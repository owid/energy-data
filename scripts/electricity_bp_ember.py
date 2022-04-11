"""Combine BP energy and Ember electricity data (global and from Europe).

"""

import argparse
import io
import os
import zipfile

import pandas as pd
import requests

from scripts import GRAPHER_DIR, INPUT_DIR
from utils import (
    add_population_to_dataframe,
    add_region_aggregates,
    list_countries_in_region_that_must_have_data,
    multi_merge,
    standardize_countries,
)

# Path to output file of combined dataset.
OUTPUT_FILE = os.path.join(GRAPHER_DIR, "Electricity mix from BP & EMBER (2022).csv")
# URL to download Ember 2022 global data from.
EMBER_GLOBAL_DATA_FILE = (
    "https://ember-climate.org/app/uploads/2022/03/Ember-GER-2022-Data.xlsx"
)
# Path to file with country names in the Ember dataset.
EMBER_GLOBAL_COUNTRIES_FILE = os.path.join(
    INPUT_DIR, "electricity-bp-ember", "ember.countries.json"
)
# URL to Ember Europe Electricity Review raw data file.
EMBER_EUROPE_DATA_FILE = (
    "https://ember-climate.org/app/uploads/2022/02/EER_2022_raw_data.zip"
)
EMBER_EUROPE_COUNTRIES_FILE = os.path.join(
    INPUT_DIR, "electricity-bp-ember", "european_electricity_review.countries.json"
)
# Path to BP energy dataset file.
BP_DATA_FILE = os.path.join(
    INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.csv"
)
BP_COUNTRIES_FILE = os.path.join(
    INPUT_DIR, "shared", "statistical_review_of_world_energy_bp_2021.countries.json"
)
# Name of files inside zip folder of European Electricity Review raw data.
EU_EMBER_GENERATION_NAME = "EER_2022_generation.csv"
EU_EMBER_EMISSIONS_NAME = "EER_2022_emissions.csv"
EU_EMBER_OVERVIEW_NAME = "EER_2022_country_overview.csv"

# Decide which dataset to prioritize if there is overlap.
# In general, we use BP as a default source of data, but we may prioritize another dataset if it is more up-to-date.
# Either "BP" to prioritise BP, or "Ember", to prioritise Ember.
BP_VS_EMBER_PRIORITY = "Ember"
# Either "EU" to prioritise data from the EER, or "Global" to prioritise data from the Global Ember electricity.
GLOBAL_VS_EU_EMBER_PRIORITY = "Global"

# Conversion factors.
# Terawatt-hours to kilowatt-hours.
TWH_TO_KWH = 1e9
# Megatonnes to grams.
MT_TO_G = 1e12

# TODO: Remove countries and regions from this blacklist once populations are consistent. In particular, the previous
#  version of the dataset considered that North America was Canada + US, which is inconsistent with current OWID
#  definition (https://ourworldindata.org/grapher/continents-according-to-our-world-in-data).
# After analysing the resulting time series, we detected several issues with the following countries/regions.
# For the moment, we remove them from the final dataset.
REGIONS_WITH_INCONSISTENT_DATA = [
    "Central America",
    "South & Central America",
    "Other South & Central America",
]

# Regions to add to each variable, following OWID definitions of those regions, instead of BP or Ember definitions.
REGIONS_TO_ADD = [
    "North America",
    "South America",
    "Europe",
    "European Union (27)",
    "Africa",
    "Asia",
    "Oceania",
    "Low-income countries",
    "Upper-middle-income countries",
    "Lower-middle-income countries",
    "High-income countries",
]


def read_csv_files_inside_remote_zip(url):
    """Read csv files that are contained inside a zip file, accessible via a given URL.
    Parameters
    ----------
    url : str
        URL pointing to zip file.
    Returns
    -------
    dataframes : dict
        Dictionary containing a dataframes for each of the csv files contained in the zip folder.
    """
    # Read a zip file (from a URL) containing .csv files.
    dataframes = {}
    # Download zip file and keep it in memory.
    r = requests.get(url)
    zip_folder = zipfile.ZipFile(io.BytesIO(r.content))
    # List files inside zip folder, ignore files that are not csv, and ignore hidden files, and read csv files.
    for name in zip_folder.namelist():
        csv_file_name = os.path.basename(name)
        if csv_file_name.endswith(".csv") and not csv_file_name.startswith("."):
            with zip_folder.open(name) as csv_file:
                df = pd.read_csv(csv_file)
                dataframes[csv_file_name] = df

    return dataframes


def load_eu_ember_generation_data():
    """Load generation data from the European Electricity Review by Ember.

    Returns
    -------
    eu_ember_elec : pd.DataFrame
        Data on electricity generation for European countries.

    """
    # Get data on electricity generation.
    columns = {
        "country_name": "Country",
        "year": "Year",
        "fuel_desc": "Variable",
        "generation_twh": "Value (TWh)",
    }
    eu_ember_elec = read_csv_files_inside_remote_zip(url=EMBER_EUROPE_DATA_FILE)[
        EU_EMBER_GENERATION_NAME
    ]
    eu_ember_elec = eu_ember_elec.rename(errors="raise", columns=columns)[
        columns.values()
    ]

    # Translate the new source groups into the old ones.
    rows = {
        "Gas": "Gas (TWh)",
        "Hydro": "Hydro (TWh)",
        "Solar": "Solar (TWh)",
        "Wind": "Wind (TWh)",
        "Bioenergy": "Bioenergy (TWh)",
        "Nuclear": "Nuclear (TWh)",
        "Other Fossil": "Oil (TWh)",
        "Other Renewables": "Other renewables excluding bioenergy (TWh)",
    }
    eu_ember_elec["Variable"] = eu_ember_elec["Variable"].replace(rows)

    # Combine the new groups Hard Coal and Lignite into the old group for Coal.
    coal_regrouped = (
        eu_ember_elec[eu_ember_elec["Variable"].isin(["Hard Coal", "Lignite"])]
        .groupby(["Country", "Year"])
        .agg({"Variable": lambda x: "Coal (TWh)", "Value (TWh)": sum})
        .reset_index()
    )

    # Remove rows with 'Hard Coal' and 'Lignite' and append the new coal groups.
    eu_ember_elec = pd.concat(
        [
            eu_ember_elec[~eu_ember_elec["Variable"].isin(["Hard Coal", "Lignite"])],
            coal_regrouped,
        ],
        ignore_index=True,
    )

    # Calculate total production.
    sources_considered = [
        "Bioenergy (TWh)",
        "Coal (TWh)",
        "Gas (TWh)",
        "Hydro (TWh)",
        "Nuclear (TWh)",
        "Oil (TWh)",
        "Other renewables excluding bioenergy (TWh)",
        "Solar (TWh)",
        "Wind (TWh)",
    ]
    total_eu_production = (
        eu_ember_elec[eu_ember_elec["Variable"].isin(sources_considered)]
        .groupby(["Country", "Year"])
        .agg({"Variable": lambda x: "Electricity generation (TWh)", "Value (TWh)": sum})
        .reset_index()
    )
    # Append total production to complete dataframe.
    eu_ember_elec = (
        pd.concat([eu_ember_elec, total_eu_production], ignore_index=True)
        .sort_values(["Country", "Year", "Variable"])
        .reset_index(drop=True)
    )

    # Reshape dataframe to have each of the individual energy sources as columns.
    eu_ember_elec = eu_ember_elec.pivot_table(
        index=["Country", "Year"], columns="Variable", values="Value (TWh)"
    ).reset_index()

    # Countries-years for which we have no data, do not appear.
    # But countries-years for which we have data about some variables (but not all) will have nan for missing variables.
    # It seems safe to assume that those data points should be zero (at least for the current dataset).
    rows_with_nan = eu_ember_elec[eu_ember_elec.isnull().any(axis=1)]
    if len(rows_with_nan) > 0:
        print(
            f"WARNING: Filling up missing data points with zero in the following cases:"
        )
        for i, row in rows_with_nan.iterrows():
            print(
                f"{row['Country']} - {row['Year']}: Missing {row[row.isnull()].index.tolist()}"
            )
    eu_ember_elec = eu_ember_elec.fillna(0)

    return eu_ember_elec


def load_eu_ember_net_imports_and_demand():
    """Load net imports and demand data from the European Electricity Review by Ember.

    Returns
    -------
    net_imports_and_demand_data : pd.DataFrame
        Data on net imports and demand for European countries.

    """
    # Get EU Ember data on net imports and electricity demand.
    net_imports_and_demand_data = read_csv_files_inside_remote_zip(
        url=EMBER_EUROPE_DATA_FILE
    )[EU_EMBER_OVERVIEW_NAME]

    # Select relevant data.
    net_imports_col = "Net imports (TWh)"
    demand_col = "Electricity demand (TWh)"
    columns = {
        "country_name": "Country",
        "year": "Year",
        "net_import_twh": net_imports_col,
        "demand_twh": demand_col,
    }
    net_imports_and_demand_data = net_imports_and_demand_data.rename(
        errors="raise", columns=columns
    )[columns.values()]

    return net_imports_and_demand_data


def load_eu_ember_emissions_data():
    """Load emissions data from the European Electricity Review by Ember.

    Returns
    -------
    eu_ember_emissions_data : pd.DataFrame
        Data on emissions for European countries.

    """
    # Get EU Ember data on net imports and electricity demand.
    eu_ember_emissions_data = read_csv_files_inside_remote_zip(
        url=EMBER_EUROPE_DATA_FILE
    )[EU_EMBER_EMISSIONS_NAME]

    columns = {
        "country_name": "Country",
        "year": "Year",
        "emissions_mtc02e": "Emissions (MtCO2)",
    }
    eu_ember_emissions_data = eu_ember_emissions_data.rename(
        errors="raise", columns=columns
    )[columns.values()]

    return eu_ember_emissions_data


def load_eu_ember_data():
    """Load and combine all datasets from the European Electricity Review data by Ember.

    Returns
    -------
    eu_ember_data : pd.DataFrame
        All Ember data for European countries.

    """
    # Load different datasets of the European Electricity Review.
    eu_ember_generation_data = load_eu_ember_generation_data()
    eu_ember_net_imports_and_demand = load_eu_ember_net_imports_and_demand()
    eu_ember_emissions_data = load_eu_ember_emissions_data()

    # Combine all datasets.
    eu_ember_data = multi_merge(
        dfs=[
            eu_ember_generation_data,
            eu_ember_net_imports_and_demand,
            eu_ember_emissions_data,
        ],
        how="outer",
        on=["Country", "Year"],
    )

    # Standardize country names.
    eu_ember_data = standardize_countries(
        df=eu_ember_data,
        countries_file=EMBER_EUROPE_COUNTRIES_FILE,
        country_col="Country",
    )
    eu_ember_data = eu_ember_data.dropna(subset="Country").reset_index(drop=True)

    return eu_ember_data


def load_global_ember_generation_data():
    """Load global electricity generation data from Ember.

    Returns
    -------
    generation : pd.DataFrame
        Global electricity generation data from Ember.

    """
    # Download global data from Ember.
    response = requests.get(EMBER_GLOBAL_DATA_FILE)
    generation = pd.read_excel(response.content, sheet_name="Generation")

    # Rename and select columns.
    columns = {
        "Country or region": "Country",
        "Year": "Year",
        "Variable": "Variable",
        "Electricity generated (TWh)": "Value (TWh)",
    }
    generation = generation.rename(errors="raise", columns=columns)[columns.values()]

    # Rename and select rows.
    rows = {
        "Total Generation": "Electricity generation (TWh)",
        "Gas": "Gas (TWh)",
        "Coal": "Coal (TWh)",
        "Other Fossil": "Oil (TWh)",
        "Nuclear": "Nuclear (TWh)",
        "Hydro": "Hydro (TWh)",
        "Solar": "Solar (TWh)",
        "Wind": "Wind (TWh)",
        "Bioenergy": "Bioenergy (TWh)",
        "Fossil": "Fossil fuels (TWh)",
        "Renewables": "Renewables (TWh)",
        "Other Renewables": "Other renewables excluding bioenergy (TWh)",
        "Clean": "Low-carbon electricity (TWh)",
        "Demand": "Electricity demand (TWh)",
        "Net Import": "Net imports (TWh)",
    }
    generation["Variable"] = generation["Variable"].replace(rows)

    generation = generation.pivot(
        index=["Country", "Year"], columns="Variable", values="Value (TWh)"
    ).reset_index()[["Country", "Year"] + list(rows.values())]

    return generation


def load_global_ember_emissions_data():
    """Load global emissions data from Ember.

    Returns
    -------
    emissions : pd.DataFrame
        Global emissions data from Ember.

    """
    # Download global data from Ember.
    response = requests.get(EMBER_GLOBAL_DATA_FILE)
    emissions = pd.read_excel(response.content, sheet_name="Emissions")

    columns = {
        "Country or region": "Country",
        "Year": "Year",
        "Emissions (MtCO2)": "Emissions (MtCO2)",
    }
    emissions = emissions.rename(errors="raise", columns=columns)[columns.values()]

    return emissions


def load_global_ember_data():
    """Load all global electricity data from Ember.

    Returns
    -------
    global_ember_data : pd.DataFrame
        Global electricity data from Ember.

    """
    # Load different datasets of global Ember electricity.
    global_ember_generation_data = load_global_ember_generation_data()
    global_ember_emissions_data = load_global_ember_emissions_data()

    # Combine datasets.
    global_ember_data = pd.merge(
        global_ember_generation_data,
        global_ember_emissions_data,
        how="outer",
        on=["Country", "Year"],
    )

    # Standardize countries, warn about countries not included in countries file, and remove them.
    global_ember_data = standardize_countries(
        df=global_ember_data,
        countries_file=EMBER_GLOBAL_COUNTRIES_FILE,
        country_col="Country",
        make_missing_countries_nan=True,
    )
    global_ember_data = global_ember_data.dropna(subset="Country").reset_index(
        drop=True
    )

    return global_ember_data


def combine_ember_data(global_ember_data, eu_ember_data):
    """Combine global and European electricity data from Ember.

    Parameters
    ----------
    global_ember_data : pd.DataFrame
        Global electricity data from Ember.
    eu_ember_data : pd.DataFrame
        European electricity data from Ember.

    Returns
    -------
    combined : pd.DataFrame
        Global and European electricity data combined.

    """
    combined = pd.concat([global_ember_data, eu_ember_data], ignore_index=True)
    if GLOBAL_VS_EU_EMBER_PRIORITY == "EU":
        combined = combined.drop_duplicates(subset=("Country", "Year"), keep="last")
    elif GLOBAL_VS_EU_EMBER_PRIORITY == "Global":
        combined = combined.drop_duplicates(subset=("Country", "Year"), keep="first")
    else:
        print(
            f"WARNING: Parameter GLOBAL_VS_EU_EMBER_PRIORITY must be either 'EU' or 'Global'."
        )

    # Add other useful aggregations.
    combined["Other renewables including bioenergy (TWh)"] = combined[
        "Other renewables excluding bioenergy (TWh)"
    ].add(combined["Bioenergy (TWh)"])
    combined["Fossil fuels (TWh)"] = (
        combined["Gas (TWh)"].add(combined["Oil (TWh)"]).add(combined["Coal (TWh)"])
    )
    combined["Renewables (TWh)"] = (
        combined["Solar (TWh)"]
        .add(combined["Wind (TWh)"])
        .add(combined["Hydro (TWh)"])
        .add(combined["Bioenergy (TWh)"])
        .add(combined["Other renewables excluding bioenergy (TWh)"])
    )
    combined["Low-carbon electricity (TWh)"] = combined["Renewables (TWh)"].add(
        combined["Nuclear (TWh)"]
    )

    return combined


def load_bp_data():
    """Load data from BP.

    Returns
    -------
    bp_elec : pd.DataFrame
        Data from BP.

    """
    columns = {
        "Entity": "Country",
        "Year": "Year",
        "Electricity Generation": "Electricity generation (TWh)",
        "Primary Energy Consumption - TWh": "Primary energy (TWh)",
        "Hydro Generation - TWh": "Hydro (TWh)",
        "Nuclear Generation - TWh": "Nuclear (TWh)",
        "Solar Generation - TWh": "Solar (TWh)",
        "Wind Generation - TWh": "Wind (TWh)",
        "Geo Biomass Other - TWh": "Other renewables including bioenergy (TWh)",
        "Elec Gen from Oil": "Oil (TWh)",
        "Elec Gen from Coal": "Coal (TWh)",
        "Elec Gen from Gas": "Gas (TWh)",
    }

    bp_elec = pd.read_csv(BP_DATA_FILE).rename(errors="raise", columns=columns)[
        columns.values()
    ]

    bp_elec["Fossil fuels (TWh)"] = (
        bp_elec["Coal (TWh)"].add(bp_elec["Oil (TWh)"]).add(bp_elec["Gas (TWh)"])
    )
    bp_elec["Renewables (TWh)"] = (
        bp_elec["Hydro (TWh)"]
        .add(bp_elec["Solar (TWh)"])
        .add(bp_elec["Wind (TWh)"])
        .add(bp_elec["Other renewables including bioenergy (TWh)"])
    )
    bp_elec["Low-carbon electricity (TWh)"] = bp_elec["Renewables (TWh)"].add(
        bp_elec["Nuclear (TWh)"]
    )

    # Ensure countries are standardized.
    bp_elec = standardize_countries(
        df=bp_elec, countries_file=BP_COUNTRIES_FILE, country_col="Country"
    )

    return bp_elec


def combine_bp_and_ember_data(bp_data, ember_data):
    """Combine data from BP and Ember.

    Parameters
    ----------
    bp_data : pd.DataFrame
        Data from BP.
    ember_data : pd.DataFrame
        Data from Ember.

    Returns
    -------
    combined : pd.DataFrame
        Combined data from BP and Ember.

    """
    bp_elec = bp_data.copy()
    ember_elec = ember_data.copy()

    # Go from wide to long format and drop NAs
    ember_elec = ember_elec.melt(id_vars=["Country", "Year"]).dropna()

    # Go from wide to long format and drop NAs
    bp_elec = bp_elec.melt(id_vars=["Country", "Year"]).dropna()

    combined = pd.concat([bp_elec, ember_elec], ignore_index=True)

    if BP_VS_EMBER_PRIORITY == "BP":
        combined = combined.drop_duplicates(
            subset=["Country", "Year", "variable"], keep="first"
        )
    else:
        combined = combined.drop_duplicates(
            subset=["Country", "Year", "variable"], keep="last"
        )

    # Go back to wide format.
    combined = (
        combined.pivot_table(
            values="value", index=["Country", "Year"], columns="variable"
        )
        .sort_values(["Country", "Year"])
        .reset_index()
    )

    return combined


def add_all_region_aggregates(df):
    """Add regions (like Europe and North America) to the dataset, following OWID definitions of those regions, and
    aggregate data for those regions appropriately.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset, which may or may not contain the regions that will be added (if contained, they will be replaced).

    Returns
    -------
    df_updated : pd.DataFrame
        Original dataset after adding regions.

    """
    df_updated = df.copy()
    informed_regions = df["Country"].unique().tolist()
    for region in REGIONS_TO_ADD:
        if region in informed_regions:
            print(f"Replacing {region} with its corresponding aggregate.")
        else:
            print(f"Adding {region}.")
        countries_that_must_have_data = list_countries_in_region_that_must_have_data(
            region=region,
            reference_year=2018,
            min_frac_individual_population=0.0,
            min_frac_cumulative_population=0.7,
        )
        df_updated = add_region_aggregates(
            df_updated,
            region,
            countries_that_must_have_data=countries_that_must_have_data,
            num_allowed_nans_per_year=None,
            frac_allowed_nans_per_year=0.2,
            country_col="Country",
            year_col="Year",
            aggregations=None,
        )

    return df_updated


def add_carbon_intensities(df):
    """Add carbon intensities to dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.

    Returns
    -------
    df : pd.DataFrame
        Dataset after adding carbon intensities.

    """
    df = df.copy()
    df["Carbon intensity of electricity (gCO2/kWh)"] = (
        df["Emissions (MtCO2)"]
        * MT_TO_G
        / (df["Electricity generation (TWh)"] * TWH_TO_KWH)
    )

    return df


def add_per_capita_variables(df):
    """Add per-capita variables to BP and Ember dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Combined data from BP and Ember without per-capita variables.

    Returns
    -------
    combined : pd.DataFrame
        Combined data from BP and Ember after adding per-capita variables.

    """
    # Add population.
    combined = add_population_to_dataframe(
        df=df, country_col="Country", year_col="Year", population_col="Population"
    )

    for cat in [
        "Electricity generation",
        "Coal",
        "Oil",
        "Gas",
        "Fossil fuels",
        "Renewables",
        "Low-carbon electricity",
        "Nuclear",
        "Solar",
        "Wind",
        "Hydro",
        "Bioenergy",
        "Other renewables excluding bioenergy",
        "Other renewables including bioenergy",
    ]:
        combined[f"{cat} electricity per capita (kWh)"] = (
            combined[f"{cat} (TWh)"] / combined["Population"] * TWH_TO_KWH
        )

    combined = combined.rename(
        errors="raise",
        columns={
            "Electricity generation electricity per capita (kWh)": "Per capita electricity (kWh)",
            "Low-carbon electricity electricity per capita (kWh)": "Low-carbon electricity per capita (kWh)",
            "Renewables electricity per capita (kWh)": "Renewable electricity per capita (kWh)",
            "Other renewables excluding bioenergy electricity per capita (kWh)": "Other renewable electricity excluding bioenergy per capita (kWh)",
            "Other renewables including bioenergy electricity per capita (kWh)": "Other renewable electricity including bioenergy per capita (kWh)",
            "Fossil fuels electricity per capita (kWh)": "Fossil fuel electricity per capita (kWh)",
        },
    )

    # Drop 'Population' column
    combined = combined.drop(errors="raise", columns=["Population"])

    return combined


def add_share_variables(df):
    """Add share variables (e.g. percentage of total electricity generated by certain sources) to BP and Ember dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Combined data from BP and Ember without share variables.

    Returns
    -------
    combined : pd.DataFrame
        Combined data from BP and Ember after adding share variables.

    """
    combined = df.copy()

    # Calculate electricity as share of energy (as a percentage).
    combined["Electricity as share of primary energy"] = (
        combined["Electricity generation (TWh)"]
        / combined["Primary energy (TWh)"]
        * 100
    )

    # Calculate the percentage of electricity demand that is imported.
    combined["Net electricity imports as a share of demand"] = (
        combined["Net imports (TWh)"] / combined["Electricity demand (TWh)"] * 100
    )

    # Calculating electricity shares by source
    for cat in [
        "Coal",
        "Oil",
        "Gas",
        "Fossil fuels",
        "Renewables",
        "Low-carbon electricity",
        "Nuclear",
        "Solar",
        "Wind",
        "Hydro",
        "Bioenergy",
        "Other renewables excluding bioenergy",
        "Other renewables including bioenergy",
    ]:
        combined[f"{cat} (% electricity)"] = (
            combined[f"{cat} (TWh)"] / combined["Electricity generation (TWh)"] * 100
        )

    return combined


def prepare_data_for_grapher(df):
    """Prepare BP & Ember dataset for grapher.

    Parameters
    ----------
    df : pd.DataFrame
        Combined dataset of BP & Ember.

    Returns
    -------
    combined : pd.DataFrame
        Combined data from BP & Ember.

    """
    combined = df.copy()

    # Remove unnecessary columns.
    combined = combined.drop(errors="raise", columns=["Primary energy (TWh)"])

    # Rename variables for grapher
    combined = combined.rename(
        errors="raise",
        columns={
            "Coal (TWh)": "Electricity from coal (TWh)",
            "Gas (TWh)": "Electricity from gas (TWh)",
            "Oil (TWh)": "Electricity from oil (TWh)",
            "Nuclear (TWh)": "Electricity from nuclear (TWh)",
            "Hydro (TWh)": "Electricity from hydro (TWh)",
            "Solar (TWh)": "Electricity from solar (TWh)",
            "Wind (TWh)": "Electricity from wind (TWh)",
            "Bioenergy (TWh)": "Electricity from bioenergy (TWh)",
            "Other renewables excluding bioenergy (TWh)": "Electricity from other renewables excluding bioenergy (TWh)",
            "Other renewables including bioenergy (TWh)": "Electricity from other renewables including bioenergy (TWh)",
            "Fossil fuels (TWh)": "Electricity from fossil fuels (TWh)",
            "Renewables (TWh)": "Electricity from renewables (TWh)",
        },
    )

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)
    combined = combined[combined.isna().sum(axis=1) < len(rounded_cols)]

    return combined


def main():
    print("Import and clean BP data")
    bp_data = load_bp_data()

    print("Import and clean global electricity data from Ember.")
    global_ember_data = load_global_ember_data()

    print("Import and clean European Electricity Review data from Ember.")
    eu_ember_data = load_eu_ember_data()

    print("Combine all data from Ember")
    ember_data = combine_ember_data(
        global_ember_data=global_ember_data, eu_ember_data=eu_ember_data
    )

    print("Combine BP and Ember data.")
    combined = combine_bp_and_ember_data(bp_data=bp_data, ember_data=ember_data)

    print("Add regions to data.")
    combined = add_all_region_aggregates(df=combined)

    print("Adding carbon intensity data from Ember.")
    combined = add_carbon_intensities(df=combined)

    print("Add per-capita variables.")
    combined = add_per_capita_variables(df=combined)

    print("Add share variables.")
    combined = add_share_variables(df=combined)

    print("Prepare data for grapher.")
    combined = prepare_data_for_grapher(df=combined)

    ####################################################################################################################
    # TODO: Remove this temporary solution once regions have been homogenized in BP dataset too.
    print(f"WARNING: Removing countries and regions with inconsistent data:")
    for region in REGIONS_WITH_INCONSISTENT_DATA:
        if region in sorted(set(combined["Country"])):
            print(f" * {region}")
    combined = combined[
        ~combined["Country"].isin(REGIONS_WITH_INCONSISTENT_DATA)
    ].reset_index(drop=True)
    ####################################################################################################################

    # Save to file as csv
    combined.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
