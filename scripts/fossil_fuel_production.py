"""Generate dataset on energy production from fossil fuels, using data from the Statistical Review of the World Energy
by BP and the Shift Dataportal.

"""

import argparse
import os
import pandas as pd
import numpy as np

from scripts import GRAPHER_DIR, INPUT_DIR
from utils import add_population_to_dataframe, standardize_countries

BP_DATA_FILE = os.path.join(
    INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.csv"
)
BP_COUNTRIES_FILE = os.path.join(
    INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.countries.json"
)
SHIFT_COAL_DATA_FILE = os.path.join(
    INPUT_DIR, "fossil-fuel-production", "shift_coal.csv"
)
SHIFT_GAS_DATA_FILE = os.path.join(INPUT_DIR, "fossil-fuel-production", "shift_gas.csv")
SHIFT_OIL_DATA_FILE = os.path.join(INPUT_DIR, "fossil-fuel-production", "shift_oil.csv")
SHIFT_COUNTRIES_FILE = os.path.join(
    INPUT_DIR, "fossil-fuel-production", "shift.countries.json"
)
OUTPUT_FILE = os.path.join(GRAPHER_DIR, "Fossil fuel production BP & Shift (2022).csv")
# Conversion factors.
# Convert exajoules to terawatt-hours.
EJ_TO_TWH = 277.778
# Convert terawatt-hours to kilowatt-hours.
TWH_TO_KWH = 1e9

# After analysing the resulting time series, we detected several issues with the following countries/regions.
# For the moment, we remove them from the final dataset.
REGIONS_WITH_INCONSISTENT_DATA = [
    # Remove North America, Central America and South & Central America, since their definitions in BP are different
    # from the definition in OWID population dataset.
    "North America",
    "Central America",
    "South & Central America",
]


def load_bp_data():
    """Load data from BP.

    Returns
    -------
    bp_data : pd.DataFrame
        Data from BP.

    """
    columns = {
        "Entity": "Country",
        "Year": "Year",
        "Coal Production - TWh": "Coal production (TWh)",
        "Gas Production - TWh": "Gas production (TWh)",
        "Oil Production - TWh": "Oil production (TWh)",
    }
    bp_data = pd.read_csv(BP_DATA_FILE, usecols=list(columns)).rename(
        errors="raise", columns=columns
    )

    bp_data = bp_data.sort_values(["Country", "Year"]).reset_index(drop=True)

    return bp_data


def load_shift_data():
    """Load data on fossil fuel production from the Shift Dataportal.

    Returns
    -------
    shift_fossil : pd.DataFrame
        Data from the Shift Dataportal.

    """
    # Load data on coal production from the Shift dataportal.
    shift_coal = pd.read_csv(SHIFT_COAL_DATA_FILE)
    shift_coal = pd.melt(
        shift_coal,
        id_vars=["Year"],
        var_name=["Country"],
        value_name="Coal Production (EJ)",
    )

    # Load data on gas production from SHIFT.
    shift_gas = pd.read_csv(SHIFT_GAS_DATA_FILE)
    shift_gas = pd.melt(
        shift_gas,
        id_vars=["Year"],
        var_name=["Country"],
        value_name="Gas Production (EJ)",
    )

    # Load data on oil production from SHIFT.
    shift_oil = pd.read_csv(SHIFT_OIL_DATA_FILE)
    shift_oil = pd.melt(
        shift_oil,
        id_vars=["Year"],
        var_name=["Country"],
        value_name="Oil Production (EJ)",
    )

    # Combine data sources.
    shift_fossil = shift_coal.merge(shift_oil, on=["Country", "Year"], how="outer")
    shift_fossil = shift_fossil.merge(shift_gas, on=["Country", "Year"], how="outer")

    # Convert units.
    shift_fossil["Coal production (TWh)"] = (
        shift_fossil["Coal Production (EJ)"] * EJ_TO_TWH
    )
    shift_fossil["Oil production (TWh)"] = (
        shift_fossil["Oil Production (EJ)"] * EJ_TO_TWH
    )
    shift_fossil["Gas production (TWh)"] = (
        shift_fossil["Gas Production (EJ)"] * EJ_TO_TWH
    )
    shift_fossil = shift_fossil.drop(
        columns=["Coal Production (EJ)", "Oil Production (EJ)", "Gas Production (EJ)"]
    )

    # Standardize countries.
    shift_fossil = standardize_countries(
        df=shift_fossil,
        countries_file=SHIFT_COUNTRIES_FILE,
        country_col="Country",
        make_missing_countries_nan=True,
    )

    # Remove missing countries and sort conveniently.
    shift_fossil = (
        shift_fossil[shift_fossil["Country"].notnull()]
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )

    return shift_fossil


def combine_bp_and_shift_data(bp_data, shift_data):
    """Combine BP and Shift data.

    Parameters
    ----------
    bp_data : pd.DataFrame
        Data from BP.
    shift_data : pd.DataFrame
        Data from Shift.

    Returns
    -------
    combined : pd.DataFrame
        Combined data from BP and Shift.

    """
    fixed_variables = ["Country", "Year"]

    # We should not concatenate bp and shift data directly, since there are nans in different places.
    variables = [col for col in bp_data.columns if col not in fixed_variables]

    combined = pd.DataFrame({fixed_variable: [] for fixed_variable in fixed_variables})

    for variable in variables:
        bp_data_for_variable = bp_data[fixed_variables + [variable]].dropna(
            subset=variable
        )
        shift_data_for_variable = shift_data[fixed_variables + [variable]].dropna(
            subset=variable
        )
        combined_for_variable = pd.concat(
            [bp_data_for_variable, shift_data_for_variable], ignore_index=True
        )
        # On rows where both datasets overlap, give priority to BP data (which is more up-to-date).
        combined_for_variable = combined_for_variable.drop_duplicates(
            subset=fixed_variables, keep="first"
        )
        # Combine data for different variables.
        combined = pd.merge(
            combined, combined_for_variable, on=fixed_variables, how="outer"
        )

    # Sort data appropriately.
    combined = combined.sort_values(fixed_variables).reset_index(drop=True)

    return combined


def add_annual_change(df):
    """Add annual change variables to combined BP & Shift dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Combined BP & Shift dataset.

    Returns
    -------
    combined : pd.DataFrame
        Combined BP & Shift dataset after adding annual change variables.

    """
    combined = df.copy()

    # Calculate annual change.
    combined = combined.sort_values(["Country", "Year"]).reset_index(drop=True)
    for cat in ("Coal", "Oil", "Gas"):
        combined[f"Annual change in {cat.lower()} production (%)"] = (
            combined.groupby("Country")[f"{cat} production (TWh)"].pct_change() * 100
        )
        combined[f"Annual change in {cat.lower()} production (TWh)"] = combined.groupby(
            "Country"
        )[f"{cat} production (TWh)"].diff()

    return combined


def add_per_capita_variables(df):
    """Add per-capita variables to combined BP & Shift dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Combined BP & Shift dataset.

    Returns
    -------
    combined : pd.DataFrame
        Combined BP & Shift dataset after adding per-capita variables.

    """
    combined = df.copy()

    # Add population to data.
    combined = add_population_to_dataframe(
        df=combined, country_col="Country", year_col="Year", population_col="Population"
    )

    # Calculate production per capita.
    for cat in ("Coal", "Oil", "Gas"):
        combined[f"{cat} production per capita (kWh)"] = (
            combined[f"{cat} production (TWh)"] / combined["Population"] * TWH_TO_KWH
        )
    combined = combined.drop(errors="raise", columns=["Population"])

    return combined


def main():
    print("Load BP data")
    bp_data = load_bp_data()

    print("Load data from Shift.")
    shift_data = load_shift_data()

    print("Combine BP and Shift data.")
    combined = combine_bp_and_shift_data(bp_data=bp_data, shift_data=shift_data)

    print("Add annual change for each source.")
    combined = add_annual_change(df=combined)

    print("Add per-capita variables.")
    combined = add_per_capita_variables(df=combined)

    print("Clean data.")
    # Remove bad data points, round data to 3 decimal places, and remove columns that only have missing values.
    combined = combined.replace([np.inf, -np.inf], np.nan)
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)
    combined = combined[combined.isna().sum(axis=1) < len(rounded_cols)]

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

    print("Save data to output file.")
    combined.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
