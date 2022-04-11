"""Generate OWID energy dataset from most up-to-date sources.

Running this script will generate the full energy dataset in three different formats:
* owid-energy-data.csv
* owid-energy-data.xlsx
* owid-energy-data.json

"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from owid import catalog

from scripts import GRAPHER_DIR, INPUT_DIR, OUTPUT_DIR
from scripts.bp_energy import main as generate_energy_mix_dataset
from scripts.electricity_bp_ember import main as generate_electricity_mix_dataset
from scripts.fossil_fuel_production import (
    main as generate_fossil_fuel_production_dataset,
)
from scripts.primary_energy_consumption import (
    main as generate_primary_energy_consumption_dataset,
)
from scripts.primary_energy_consumption import load_maddison_data

# Define paths to output files.
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, "owid-energy-data.csv")
OUTPUT_EXCEL_FILE = os.path.join(OUTPUT_DIR, "owid-energy-data.xlsx")
OUTPUT_JSON_FILE = os.path.join(OUTPUT_DIR, "owid-energy-data.json")
# Define paths to input files.
PRIMARY_ENERGY_CONSUMPTION_FILE = os.path.join(
    GRAPHER_DIR, "Primary energy consumption BP & EIA (2022).csv"
)
BP_ENERGY_FILE = os.path.join(
    INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.csv"
)
ENERGY_MIX_FROM_BP_FILE = os.path.join(GRAPHER_DIR, "Energy mix from BP (2021).csv")
ELECTRICITY_MIX_FROM_BP_AND_EMBER_FILE = os.path.join(
    GRAPHER_DIR, "Electricity mix from BP & EMBER (2022).csv"
)
FOSSIL_FUEL_PRODUCTION_FILE = os.path.join(
    GRAPHER_DIR, "Fossil fuel production BP & Shift (2022).csv"
)
# TODO: Remove this file once full population dataset can be loaded from OWID catalog.
POPULATION_FILE = os.path.join(INPUT_DIR, "shared", "population.csv")


def df_to_json(complete_dataset, output_path, static_columns):
    megajson = {}

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


def load_population_with_iso_codes():
    # Load population data and calculate per capita energy.
    population = (
        catalog.find("population", namespace="owid", dataset="key_indicators")
        .load()
        .reset_index()
        .rename(
            columns={
                "country": "Country",
                "year": "Year",
                "population": "Population",
            }
        )[["Country", "Year", "Population"]]
    )

    ####################################################################################################################
    # TODO: Remove temporary solution once OWID population dataset is complete.
    additional_population = pd.read_csv(POPULATION_FILE)
    population = pd.concat(
        [population, additional_population], ignore_index=True
    ).drop_duplicates(subset=["Country", "Year"], keep="first")
    ####################################################################################################################
    population = population.sort_values(["Country", "Year"]).reset_index(drop=True)

    # Load OWID countries_regions dataset.
    countries_regions = catalog.find(
        "countries_regions", dataset="reference", namespace="owid"
    ).load()
    countries_regions = countries_regions.reset_index().rename(
        columns={"name": "Country", "code": "iso_code"}
    )[["Country", "iso_code"]]

    # Add iso codes to population dataframe.
    population = pd.merge(population, countries_regions, on=["Country"], how="left")

    return population


def load_bp_data():
    columns = {
        "Entity": "Country",
        "Year": "Year",
        "Coal Consumption - TWh": "Coal Consumption - TWh",
        "Oil Consumption - TWh": "Oil Consumption - TWh",
        "Gas Consumption - TWh": "Gas Consumption - TWh",
    }
    bp_data = pd.read_csv(BP_ENERGY_FILE).rename(errors="raise", columns=columns)[
        columns.values()
    ]

    return bp_data


def generate_combined_dataset():
    # Add Primary Energy Consumption
    primary_energy = pd.read_csv(PRIMARY_ENERGY_CONSUMPTION_FILE)

    # Add BP data for Coal, Oil and Gas consumption
    bp_energy = load_bp_data()

    # Add Energy Mix
    energy_mix = pd.read_csv(ENERGY_MIX_FROM_BP_FILE)

    # Add Electricity Mix
    elec_mix = pd.read_csv(ELECTRICITY_MIX_FROM_BP_AND_EMBER_FILE)

    # Add Fossil Fuel Production
    fossil_fuels = pd.read_csv(FOSSIL_FUEL_PRODUCTION_FILE)

    # Add population and GDP data
    population = load_population_with_iso_codes()
    gdp = load_maddison_data()

    # merges together energy datasets
    combined = (
        primary_energy.merge(
            fossil_fuels, on=["Year", "Country"], how="outer", validate="1:1"
        )
        .merge(bp_energy, on=["Year", "Country"], how="outer", validate="1:1")
        .merge(energy_mix, on=["Year", "Country"], how="outer", validate="1:1")
        .merge(elec_mix, on=["Year", "Country"], how="outer", validate="1:1")
    )

    combined = combined.drop(
        errors="raise",
        columns=[
            "Biofuels (% primary direct energy)",
            "Coal (% primary direct energy)",
            "Electricity as share of primary energy",
            "Fossil Fuels (% primary direct energy)",
            "Gas (% primary direct energy)",
            "Hydro (% primary direct energy)",
            "Low-carbon energy (% primary direct energy)",
            "Low-carbon energy (TWh)",
            "Nuclear (% primary direct energy)",
            "Oil (% primary direct energy)",
            "Other renewables (% primary direct energy)",
            "Primary energy (TWh)",
            "Primary energy – direct (TWh)",
            "Renewables (% primary direct energy)",
            "Solar (% primary direct energy)",
            "Wind (% primary direct energy)",
        ],
    )

    row_has_data = combined.drop(columns=["Country", "Year"]).notnull().any(axis=1)
    countries_keep = set(primary_energy["Country"].unique())
    combined = combined[row_has_data & combined["Country"].isin(countries_keep)]

    # merges non-energy datasets onto energy dataset
    combined = combined.merge(
        population, on=["Year", "Country"], how="left", validate="1:1"
    ).merge(gdp, on=["Year", "Country"], how="left", validate="1:1")

    # Reorder columns
    left_columns = ["iso_code", "Country", "Year"]
    other_columns = sorted([col for col in combined.columns if col not in left_columns])
    column_order = left_columns + other_columns
    combined = combined[column_order]

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)
    combined = combined[combined.isna().sum(axis=1) < len(rounded_cols)]

    # Convert to machine-readable format
    combined = combined.rename(
        errors="raise",
        columns={
            "Country": "country",
            "Year": "year",
            "Annual change in coal production (%)": "coal_prod_change_pct",
            "Annual change in coal production (TWh)": "coal_prod_change_twh",
            "Annual change in gas production (%)": "gas_prod_change_pct",
            "Annual change in gas production (TWh)": "gas_prod_change_twh",
            "Annual change in oil production (%)": "oil_prod_change_pct",
            "Annual change in oil production (TWh)": "oil_prod_change_twh",
            "Annual change primary energy consumption (%)": "energy_cons_change_pct",
            "Annual change primary energy consumption (TWh)": "energy_cons_change_twh",
            "Bioenergy (% electricity)": "biofuel_share_elec",
            "Biofuels (% growth)": "biofuel_cons_change_pct",
            "Biofuels (% sub energy)": "biofuel_share_energy",
            "Biofuels (TWh growth – sub method)": "biofuel_cons_change_twh",
            "Bioenergy electricity per capita (kWh)": "biofuel_elec_per_capita",
            "Biofuels (TWh)": "biofuel_consumption",
            "Biofuels per capita (kWh)": "biofuel_cons_per_capita",
            "Carbon intensity of electricity (gCO2/kWh)": "carbon_intensity_elec",
            "Coal Consumption - TWh": "coal_consumption",
            "Coal (% electricity)": "coal_share_elec",
            "Coal (% growth)": "coal_cons_change_pct",
            "Coal (% sub energy)": "coal_share_energy",
            "Coal (TWh growth – sub method)": "coal_cons_change_twh",
            "Coal electricity per capita (kWh)": "coal_elec_per_capita",
            "Coal per capita (kWh)": "coal_cons_per_capita",
            "Coal production (TWh)": "coal_production",
            "Coal production per capita (kWh)": "coal_prod_per_capita",
            "Electricity generation (TWh)": "electricity_generation",
            "Emissions (MtCO2)": "greenhouse_gas_emissions",
            "Electricity from bioenergy (TWh)": "biofuel_electricity",
            "Electricity from coal (TWh)": "coal_electricity",
            "Electricity from fossil fuels (TWh)": "fossil_electricity",
            "Electricity from gas (TWh)": "gas_electricity",
            "Electricity from hydro (TWh)": "hydro_electricity",
            "Electricity from nuclear (TWh)": "nuclear_electricity",
            "Electricity from oil (TWh)": "oil_electricity",
            "Electricity from other renewables including bioenergy (TWh)": "other_renewable_electricity",
            "Electricity from other renewables excluding bioenergy (TWh)": "other_renewable_exc_biofuel_electricity",
            "Electricity from renewables (TWh)": "renewables_electricity",
            "Electricity from solar (TWh)": "solar_electricity",
            "Electricity from wind (TWh)": "wind_electricity",
            "Energy per GDP (kWh per $)": "energy_per_gdp",
            "Energy per capita (kWh)": "energy_per_capita",
            "Fossil Fuels (TWh)": "fossil_fuel_consumption",
            "Fossil fuel electricity per capita (kWh)": "fossil_cons_per_capita",
            "Fossil fuels (% electricity)": "fossil_share_elec",
            "Fossil Fuels (% growth)": "fossil_cons_change_pct",
            "Fossil Fuels (% sub energy)": "fossil_share_energy",
            "Fossil Fuels (TWh growth – sub method)": "fossil_cons_change_twh",
            "Fossil Fuels per capita (kWh)": "fossil_energy_per_capita",
            "Gas Consumption - TWh": "gas_consumption",
            "Gas (% electricity)": "gas_share_elec",
            "Gas (% growth)": "gas_cons_change_pct",
            "Gas (% sub energy)": "gas_share_energy",
            "Gas (TWh growth – sub method)": "gas_cons_change_twh",
            "Gas electricity per capita (kWh)": "gas_elec_per_capita",
            "Gas per capita (kWh)": "gas_energy_per_capita",
            "Gas production (TWh)": "gas_production",
            "Gas production per capita (kWh)": "gas_prod_per_capita",
            "Hydro (% electricity)": "hydro_share_elec",
            "Hydro (% growth)": "hydro_cons_change_pct",
            "Hydro (% sub energy)": "hydro_share_energy",
            "Hydro (TWh growth – sub method)": "hydro_cons_change_twh",
            "Hydro (TWh – sub method)": "hydro_consumption",
            "Hydro electricity per capita (kWh)": "hydro_elec_per_capita",
            "Hydro per capita (kWh)": "hydro_energy_per_capita",
            "Low-carbon electricity (% electricity)": "low_carbon_share_elec",
            "Low-carbon electricity (TWh)": "low_carbon_electricity",
            "Low-carbon electricity per capita (kWh)": "low_carbon_elec_per_capita",
            "Low-carbon energy (% growth)": "low_carbon_cons_change_pct",
            "Low-carbon energy (% sub energy)": "low_carbon_share_energy",
            "Low-carbon energy (TWh growth – sub method)": "low_carbon_cons_change_twh",
            "Low-carbon energy (TWh – sub method)": "low_carbon_consumption",
            "Low-carbon energy per capita (kWh)": "low_carbon_energy_per_capita",
            "Nuclear (% electricity)": "nuclear_share_elec",
            "Nuclear (% growth)": "nuclear_cons_change_pct",
            "Nuclear (% sub energy)": "nuclear_share_energy",
            "Nuclear (TWh growth – sub method)": "nuclear_cons_change_twh",
            "Nuclear (TWh – sub method)": "nuclear_consumption",
            "Nuclear electricity per capita (kWh)": "nuclear_elec_per_capita",
            "Nuclear per capita (kWh)": "nuclear_energy_per_capita",
            "Oil Consumption - TWh": "oil_consumption",
            "Oil (% electricity)": "oil_share_elec",
            "Oil (% growth)": "oil_cons_change_pct",
            "Oil (% sub energy)": "oil_share_energy",
            "Oil (TWh growth – sub method)": "oil_cons_change_twh",
            "Oil electricity per capita (kWh)": "oil_elec_per_capita",
            "Oil per capita (kWh)": "oil_energy_per_capita",
            "Oil production (TWh)": "oil_production",
            "Oil production per capita (kWh)": "oil_prod_per_capita",
            "Other renewable electricity including bioenergy per capita (kWh)": "other_renewables_elec_per_capita",
            "Other renewable electricity excluding bioenergy per capita (kWh)": "other_renewables_elec_per_capita_exc_biofuel",
            "Other renewables including bioenergy (% electricity)": "other_renewables_share_elec",
            "Other renewables excluding bioenergy (% electricity)": "other_renewables_share_elec_exc_biofuel",
            "Other renewables (% growth)": "other_renewables_cons_change_pct",
            "Other renewables (% sub energy)": "other_renewables_share_energy",
            "Other renewables (TWh growth – sub method)": "other_renewables_cons_change_twh",
            "Other renewables (TWh – sub method)": "other_renewable_consumption",
            "Other renewables per capita (kWh)": "other_renewables_energy_per_capita",
            "Per capita electricity (kWh)": "per_capita_electricity",
            "Population": "population",
            "Primary energy consumption (TWh)": "primary_energy_consumption",
            "Renewables (% electricity)": "renewables_share_elec",
            "Renewables (% growth)": "renewables_cons_change_pct",
            "Renewables (% sub energy)": "renewables_share_energy",
            "Renewables (TWh growth – sub method)": "renewables_cons_change_twh",
            "Renewables (TWh – sub method)": "renewables_consumption",
            "Renewables per capita (kWh)": "renewables_energy_per_capita",
            "Renewable electricity per capita (kWh)": "renewables_elec_per_capita",
            "Solar (% electricity)": "solar_share_elec",
            "Solar (% growth)": "solar_cons_change_pct",
            "Solar (% sub energy)": "solar_share_energy",
            "Solar (TWh growth – sub method)": "solar_cons_change_twh",
            "Solar (TWh – sub method)": "solar_consumption",
            "Solar electricity per capita (kWh)": "solar_elec_per_capita",
            "Solar per capita (kWh)": "solar_energy_per_capita",
            "GDP": "gdp",
            "Wind (% electricity)": "wind_share_elec",
            "Wind (% growth)": "wind_cons_change_pct",
            "Wind (% sub energy)": "wind_share_energy",
            "Wind (TWh growth – sub method)": "wind_cons_change_twh",
            "Wind (TWh – sub method)": "wind_consumption",
            "Wind electricity per capita (kWh)": "wind_elec_per_capita",
            "Wind per capita (kWh)": "wind_energy_per_capita",
            "Net imports (TWh)": "net_elec_imports",
            "Electricity demand (TWh)": "electricity_demand",
            "Net electricity imports as a share of demand": "net_elec_imports_share_demand",
        },
    )

    combined.sort_values(["country", "year"], inplace=True)

    # Remove spurious infinity values (which should be further investigated during sanity checks).
    for column in combined.columns:
        issues_mask = combined[column] == np.inf
        issues = combined[issues_mask]
        if len(issues) > 0:
            print(
                f"WARNING: Removing {len(issues)} infinity values found in '{column}'. Affected countries:"
            )
            for country in set(issues["country"]):
                print(f"* {country}")
            combined.loc[issues_mask, column] = np.nan

    # Save output files
    combined.to_csv(OUTPUT_CSV_FILE, index=False)
    combined.to_excel(OUTPUT_EXCEL_FILE, index=False)
    df_to_json(combined, OUTPUT_JSON_FILE, ["iso_code"])


def main():
    print("\nGenerating energy mix dataset.")
    generate_energy_mix_dataset()

    print("\nGenerating electricity mix dataset.")
    generate_electricity_mix_dataset()

    print("\nGenerating fossil fuel energy production dataset.")
    generate_fossil_fuel_production_dataset()

    print("\nGenerating primary energy consumption dataset.")
    generate_primary_energy_consumption_dataset()

    print("\nGenerating energy dataset.")
    generate_combined_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
