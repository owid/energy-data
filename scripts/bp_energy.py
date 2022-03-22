"""Generate energy mix dataset using data from BP's statistical review.

"""

import argparse
import os

import numpy as np
import pandas as pd
from owid import catalog

from scripts import GRAPHER_DIR, INPUT_DIR

# Original BP statistical review can be accessed at
# https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy.html
# The data has been processed by OWID to generate a dataset with a convenient choice of variables and units.
# The code for this processing can be found in
# https://github.com/owid/importers/tree/master/bp_statreview
# TODO: As a temporary solution, the Statistical Review of the World Energy by BP (processed in importers repository)
#  has been downloaded as a csv file and added here. Once that dataset is in owid catalog, remove this file.
BP_INPUT_FILE = os.path.join(INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.csv")
# Output file.
BP_OUTPUT_FILE = os.path.join(GRAPHER_DIR, "Energy mix from BP (2021).csv")

# Conversion factors.
# Terawatt-hours to kilowatt-hours.
TWH_TO_KWH = 1e9
# Exajoules to terawatt-hours.
EJ_TO_TWH = 277.778
# Petajoules to exajoules.
PJ_TO_EJ = 1e-3


def main():
    bp_data = pd.read_csv(BP_INPUT_FILE, usecols=[
        "Entity",
        "Year",
        "Coal Consumption - EJ",
        "Gas Consumption - EJ",
        "Oil Consumption - EJ",
        "Hydro Consumption - EJ",
        "Nuclear Consumption - EJ",
        "Biofuels Consumption - PJ - Total",
        "Primary Energy Consumption - TWh",
        "Solar Consumption - EJ",
        "Wind Consumption - EJ",
        "Geo Biomass Other - EJ",
        "Hydro Generation - TWh",
        "Nuclear Generation - TWh",
        "Solar Generation - TWh",
        "Wind Generation - TWh",
        "Geo Biomass Other - TWh",
            ])

    primary_energy = bp_data.rename(
        errors="raise",
        columns={
            "Coal Consumption - EJ": "Coal (EJ)",
            "Gas Consumption - EJ": "Gas (EJ)",
            "Oil Consumption - EJ": "Oil (EJ)",
            "Hydro Consumption - EJ": "Hydro (EJ)",
            "Nuclear Consumption - EJ": "Nuclear (EJ)",
            "Solar Consumption - EJ": "Solar (EJ)",
            "Wind Consumption - EJ": "Wind (EJ)",
            "Geo Biomass Other - EJ": "Other renewables (EJ)",
            "Primary Energy Consumption - TWh": "Primary Energy (EJ)",
            "Entity": "Country",
            "Hydro Generation - TWh": "Hydro (TWh)",
            "Nuclear Generation - TWh": "Nuclear (TWh)",
            "Solar Generation - TWh": "Solar (TWh)",
            "Wind Generation - TWh": "Wind (TWh)",
            "Geo Biomass Other - TWh": "Other renewables (TWh)",
            "Biofuels Consumption - PJ - Total": "Biofuels (PJ)",
        },
    )

    primary_energy["Biofuels (EJ)"] = primary_energy["Biofuels (PJ)"] * PJ_TO_EJ

    primary_energy["Fossil Fuels (EJ)"] = (
        primary_energy["Coal (EJ)"]
        .add(primary_energy["Oil (EJ)"])
        .add(primary_energy["Gas (EJ)"])
    )
    # To avoid many missing values in total renewable energy, assume missing values in Biofuels mean zero consumption.
    # By visually inspecting the original data, this seems to be a reasonable assumption: most missing values .
    primary_energy["Renewables (EJ)"] = (
        primary_energy["Hydro (EJ)"]
        .add(primary_energy["Solar (EJ)"])
        .add(primary_energy["Wind (EJ)"])
        .add(primary_energy["Other renewables (EJ)"])
        .add(primary_energy["Biofuels (EJ)"].fillna(0))
    )
    primary_energy["Low-carbon energy (EJ)"] = primary_energy["Renewables (EJ)"].add(
        primary_energy["Nuclear (EJ)"]
    )

    # Converting all sources to TWh (primary energy – sub method)
    for cat in ["Coal", "Oil", "Gas", "Biofuels"]:
        primary_energy[f"{cat} (TWh)"] = primary_energy[f"{cat} (EJ)"] * EJ_TO_TWH

    for cat in [
        "Hydro",
        "Nuclear",
        "Renewables",
        "Solar",
        "Wind",
        "Other renewables",
        "Low-carbon energy",
    ]:
        primary_energy[f"{cat} (TWh – sub method)"] = (
            primary_energy[f"{cat} (EJ)"] * EJ_TO_TWH
        )

    primary_energy["Renewables (TWh)"] = (
        primary_energy["Hydro (TWh)"]
        .add(primary_energy["Solar (TWh)"])
        .add(primary_energy["Wind (TWh)"])
        .add(primary_energy["Other renewables (TWh)"])
        .add(primary_energy["Biofuels (TWh)"])
    )
    primary_energy["Low-carbon energy (TWh)"] = primary_energy["Renewables (TWh)"].add(
        primary_energy["Nuclear (TWh)"]
    )
    primary_energy["Fossil Fuels (TWh)"] = (
        primary_energy["Coal (TWh)"]
        .add(primary_energy["Oil (TWh)"])
        .add(primary_energy["Gas (TWh)"])
    )
    primary_energy["Primary energy (TWh)"] = primary_energy["Fossil Fuels (TWh)"].add(
        primary_energy["Low-carbon energy (TWh – sub method)"]
    )

    # Calculating each source as share of direct primary energy
    primary_energy["Primary energy – direct (TWh)"] = (
        primary_energy["Fossil Fuels (TWh)"] + primary_energy["Low-carbon energy (TWh)"]
    )

    for cat in [
        "Coal",
        "Gas",
        "Oil",
        "Biofuels",
        "Nuclear",
        "Hydro",
        "Renewables",
        "Solar",
        "Wind",
        "Other renewables",
        "Fossil Fuels",
        "Low-carbon energy",
    ]:
        primary_energy[f"{cat} (% primary direct energy)"] = (
            primary_energy[f"{cat} (TWh)"]
            / primary_energy["Primary energy – direct (TWh)"]
            * 100
        )

    # Calculating each source as share of energy (substitution method)
    for cat in [
        "Coal",
        "Gas",
        "Oil",
        "Biofuels",
        "Nuclear",
        "Hydro",
        "Renewables",
        "Solar",
        "Wind",
        "Other renewables",
        "Fossil Fuels",
        "Low-carbon energy",
    ]:
        primary_energy[f"{cat} (% sub energy)"] = (
            primary_energy[f"{cat} (EJ)"] / primary_energy["Primary Energy (EJ)"] * 100
        )

    # Calculating annual change in each source
    primary_energy = primary_energy.sort_values(["Country", "Year"])

    for cat in ["Coal", "Oil", "Gas", "Biofuels", "Fossil Fuels"]:
        primary_energy[f"{cat} (% growth)"] = (
            primary_energy.groupby("Country")[f"{cat} (TWh)"].pct_change() * 100
        )
        primary_energy[f"{cat} (TWh growth – sub method)"] = primary_energy.groupby(
            "Country"
        )[f"{cat} (TWh)"].diff()

    for cat in [
        "Hydro",
        "Nuclear",
        "Renewables",
        "Solar",
        "Wind",
        "Other renewables",
        "Low-carbon energy",
    ]:
        primary_energy[f"{cat} (% growth)"] = (
            primary_energy.groupby("Country")[f"{cat} (TWh – sub method)"].pct_change()
            * 100
        )
        primary_energy[f"{cat} (TWh growth – sub method)"] = primary_energy.groupby(
            "Country"
        )[f"{cat} (TWh – sub method)"].diff()

    # Load population data and calculate per capita energy.
    population = catalog.find("population", namespace="owid", dataset="key_indicators").load().reset_index().rename(
            columns={"country": "Country", "year": "Year", "population": "Population"}
    )[["Country", "Year", "Population"]]
    # Check if there is any missing country.
    missing_countries = set(primary_energy['Country']) - set(population['Country'])
    if len(missing_countries) > 0:
        print(f"WARNING: {len(missing_countries)} countries not found in population dataset:.")
        print("  They will remain in the dataset, but have no population data.")
        print('\n'.join(missing_countries))
    primary_energy = primary_energy.merge(population, on=["Country", "Year"])

    for cat in ["Coal", "Oil", "Gas", "Biofuels", "Fossil Fuels"]:
        primary_energy[f"{cat} per capita (kWh)"] = (
            primary_energy[f"{cat} (TWh)"] / primary_energy["Population"] * TWH_TO_KWH
        )

    for cat in [
        "Hydro",
        "Nuclear",
        "Renewables",
        "Solar",
        "Wind",
        "Other renewables",
        "Low-carbon energy",
    ]:
        primary_energy[f"{cat} per capita (kWh)"] = (
            primary_energy[f"{cat} (TWh – sub method)"]
            / primary_energy["Population"]
            * TWH_TO_KWH
        )

    energy_mix = primary_energy.drop(
        errors="raise",
        columns=[
            "Coal (EJ)",
            "Gas (EJ)",
            "Oil (EJ)",
            "Biofuels (PJ)",
            "Biofuels (EJ)",
            "Hydro (EJ)",
            "Nuclear (EJ)",
            "Renewables (EJ)",
            "Solar (EJ)",
            "Wind (EJ)",
            "Other renewables (EJ)",
            "Primary Energy (EJ)",
            "Fossil Fuels (EJ)",
            "Low-carbon energy (EJ)",
            "Hydro (TWh)",
            "Nuclear (TWh)",
            "Solar (TWh)",
            "Wind (TWh)",
            "Other renewables (TWh)",
            "Renewables (TWh)",
            "Coal (TWh)",
            "Oil (TWh)",
            "Gas (TWh)",
            "Population",
        ],
    )

    energy_mix = energy_mix.replace([np.inf, -np.inf], np.nan)

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(energy_mix) if col not in ("Country", "Year")]
    energy_mix[rounded_cols] = energy_mix[rounded_cols].round(3)
    energy_mix = energy_mix[energy_mix.isna().sum(axis=1) < len(rounded_cols)]

    # Save to files as csv
    energy_mix.to_csv(BP_OUTPUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
