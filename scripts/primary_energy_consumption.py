"""Load BP and EIA data on primary energy consumption, combine both datasets, add variables, and export as a csv file.

"""

import argparse
import os

import pandas as pd

from scripts import GRAPHER_DIR, INPUT_DIR
from utils import add_population_to_dataframe, standardize_countries

# Input data files.
# BP data file.
# TODO: Use updated dataset (statistical review).
BP_DATA_FILE = os.path.join(INPUT_DIR, "shared", "bp_energy.csv")
# EIA data file, manually downloaded from
# https://www.eia.gov/international/data/world/total-energy/more-total-energy-data
# after selecting all countries, only "Consumption" in activities, "MTOE" in units, and downloaded as CSV (table).
EIA_DATA_FILE = os.path.join(INPUT_DIR, "energy-consumption", "eia_primary_energy_consumption.csv")
# EIA population file, mapping EIA country names to OWID country names, generated using the etl.harmonize tool.
EIA_POPULATION_FILE = os.path.join(INPUT_DIR, "energy-consumption", "eia.countries.json")
# GDP Maddison file.
# TODO: Instead of loading it from a file, import it from owid catalog.
GDP_MADDISON_FILE = os.path.join(INPUT_DIR, "shared", "total-gdp-maddison.csv")
# Output file of combined BP & EIA data.
OUTPUT_FILE = os.path.join(GRAPHER_DIR, "Primary energy consumption BP & EIA (2022).csv")

# Conversion factors.
# Million tonnes of oil equivalent to terawatt-hours.
MTOE_TO_TWH = 11.63
# Exajoules to terawatt-hours.
EJ_TO_TWH = 277.778
# Terawatt-hours to kilowatt-hours.
TWH_TO_KWH = 1e9

# After analysing the resulting time series, we detected several issues with the following countries/regions.
# For the moment, we remove them from the final dataset.
REGIONS_WITH_INCONSISTENT_DATA = [
    "Gibraltar",
]


def load_eia_data():
    """Load EIA data on primary energy consumption.

    Returns
    -------
    eia_data : pd.DataFrame
        EIA data.

    """
    eia_data = pd.read_csv(EIA_DATA_FILE, skiprows=1, na_values='--').drop(columns=['API']).\
        rename(columns={'Unnamed: 1': 'Country'})
    assert eia_data.iloc[0, 0] == 'total energy consumption (MMTOE)'
    eia_data = eia_data.iloc[1:].reset_index(drop=True)

    # Adjust the format.
    col_name = "Primary energy consumption (TWh)"
    eia_data = eia_data.melt(id_vars='Country', var_name='Year', value_name=col_name)
    eia_data["Year"] = eia_data["Year"].astype(int)
    # Convert units.
    eia_data[col_name] = eia_data[col_name] * MTOE_TO_TWH
    # Remove appended spaces on country names.
    eia_data["Country"] = eia_data["Country"].str.lstrip()

    # Standardize country names.
    eia_data = standardize_countries(df=eia_data, countries_file=EIA_POPULATION_FILE, country_col='Country',
                                     warn_on_missing_countries=True, make_missing_countries_nan=False,
                                     warn_on_unused_countries=True, show_full_warning=True)

    # Drop rows with missing values and sort rows conveniently.
    eia_data = eia_data.dropna(subset=[col_name]).sort_values(['Country', 'Year']).reset_index(drop=True)

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
        "Primary Energy Consumption": "Primary energy consumption (EJ)",
    }
    # Import total primary energy consumption data from BP.
    bp_data = pd.read_csv(BP_DATA_FILE, usecols=list(columns)).rename(errors="raise", columns=columns)

    # Convert units.
    bp_data["Primary energy consumption (TWh)"] = bp_data["Primary energy consumption (EJ)"] * EJ_TO_TWH
    bp_data = bp_data.drop(errors="raise", columns=["Primary energy consumption (EJ)"])

    # Drop rows with missing values and sort rows conveniently.
    bp_data = bp_data.dropna(subset="Primary energy consumption (TWh)").sort_values(['Country', 'Year']).\
        reset_index(drop=True)

    return bp_data


def main():
    # Load BP data and EIA data on primary energy consumption.
    bp_data = load_bp_data()
    eia_data = load_eia_data()

    # Combine energy consumption from BP and EIA, and prioritise the former.
    combined = pd.concat([bp_data, eia_data], ignore_index=True).\
        drop_duplicates(subset=['Country', 'Year'], keep="first").sort_values(["Country", "Year"]).\
        reset_index(drop=True)

    # Calculate annual change.
    combined["Annual change primary energy consumption (%)"] = (
        combined.groupby("Country")["Primary energy consumption (TWh)"].pct_change()
        * 100
    )
    combined["Annual change primary energy consumption (TWh)"] = combined.groupby(
        "Country"
    )["Primary energy consumption (TWh)"].diff()

    # Add population to primary energy dataframe
    combined = add_population_to_dataframe(df=combined, country_col='Country', year_col='Year',
                                           population_col='Population', warn_on_missing_countries=True,
                                           show_full_warning=True)

    combined["Energy per capita (kWh)"] = (
        combined["Primary energy consumption (TWh)"]
        / combined["Population"]
        * TWH_TO_KWH
    )
    combined = combined.drop(errors="raise", columns=["Population"])

    # Load GDP data and calculate energy consumption per unit GDP.
    gdp = pd.read_csv(GDP_MADDISON_FILE, usecols=["Country", "Year", "Total real GDP"])
    combined = combined.merge(gdp, on=["Country", "Year"], how="left")
    combined["Energy per GDP (kWh per $)"] = (
        combined["Primary energy consumption (TWh)"]
        / combined["Total real GDP"]
        * TWH_TO_KWH
    )
    combined = combined.drop(errors="raise", columns=["Total real GDP"])

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)    

    print(f"Removing countries and regions with inconsistent data:")
    for region in REGIONS_WITH_INCONSISTENT_DATA:
        print(f" * {region}")
    combined = combined[
        ~combined["Country"].isin(REGIONS_WITH_INCONSISTENT_DATA)
    ].reset_index(drop=True)

    # Remove rows without any relevant data and sort conveniently.
    combined = combined.dropna(subset=rounded_cols, how='all').sort_values(['Country', 'Year']).reset_index(drop=True)

    # Save data to file.
    combined.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
