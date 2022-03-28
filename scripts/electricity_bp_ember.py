"""Combine BP energy and Ember electricity data (global and from Europe).

"""

import argparse
import io
import json
import os
import zipfile

import pandas as pd
import requests
from owid import catalog

from scripts import GRAPHER_DIR, INPUT_DIR
from utils import add_population_to_dataframe, standardize_countries

# Path to output file of combined dataset.
OUTPUT_FILE = os.path.join(GRAPHER_DIR, "Electricity mix from BP & EMBER (2022).csv")
# URL to download Ember 2021 global data from.
EMBER_GLOBAL_DATA_FILE = "https://ember-climate.org/app/uploads/2021/03/Data-Global-Electricity-Review-2021.xlsx"
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
# Carbon intensity is also included in the EER (including 2021), however they do not include UK.
# This is data from the previous release (which includes UK, although only until 2020).
EMBER_EUROPE_CARBON_INTENSITY_FILE = os.path.join(
    INPUT_DIR, "electricity-bp-ember", "eu_electricity_ember.xlsx"
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
BP_VS_EMBER_PRIORITY = "Ember"
GLOBAL_VS_EU_EMBER_PRIORITY = "EU"

# Conversion factors.
# Terawatt-hours to kilowatt-hours.
TWH_TO_KWH = 1e9

# TODO: Remove countries and regions from this blacklist once populations are consistent. In particular, the previous
#  version of the dataset considered that North America was Canada + US, which is inconsistent with current OWID
#  definition (https://ourworldindata.org/grapher/continents-according-to-our-world-in-data).
# After analysing the resulting time series, we detected several issues with the following countries/regions.
# For the moment, we remove them from the final dataset.
REGIONS_WITH_INCONSISTENT_DATA = [
    "Central America",
    "Moldova",
    "North America",
    "South & Central America",
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


def load_eu_27_countries():
    """Load list of 27 countries in the European Union.

    Returns
    -------
    eu_countries : list
        EU (27) countries.

    """
    # Load OWID countries regions dataset.
    countries_regions = catalog.find(
        table="countries_regions", dataset="reference", namespace="owid"
    ).load()

    # Get list of countries in the EU.
    eu_countries = json.loads(countries_regions.loc["OWID_EU27"]["members"])

    return eu_countries


def load_european_electricity_review_data():
    """Load and prepare European Electricity Review (EER) electricity data from Ember in a convenient format.

    Generate a dataframe analogous to the one for Ember global electricity, but for European countries, based on the
    European Electricity Review from Ember. Changes to original EER file from Ember:
    * Add 'Hard Coal' to 'Lignite' and call it simply 'Coal'.

    Returns
    -------
    eu_ember_elec : pd.DataFrame
        Ember electricity data for Europe.

    """
    # Get data on electricity generation.
    eu_ember_elec = read_csv_files_inside_remote_zip(url=EMBER_EUROPE_DATA_FILE)[
        EU_EMBER_GENERATION_NAME
    ]

    # TODO: Instead of doing this, export to csv the names of countries and follow the harmonize process.

    # Change columns for consistency with ember_elec dataframe.
    # I assume that the country code is ISO 3.
    eu_ember_elec = eu_ember_elec[
        ["country_code", "year", "fuel_desc", "generation_twh"]
    ].rename(
        errors="raise",
        columns={
            "country_code": "iso_alpha3",
            "year": "Year",
            "fuel_desc": "Variable",
            "generation_twh": "Value (TWh)",
        },
    )

    # Standardize country names.
    countries_regions = (
        catalog.find("countries_regions", namespace="owid").load().reset_index()
    )

    eu_ember_elec = pd.merge(
        eu_ember_elec,
        countries_regions[["iso_alpha3", "name"]],
        on="iso_alpha3",
        how="left",
    ).rename(errors="raise", columns={"name": "Country"})[
        ["Country", "Year", "Variable", "Value (TWh)", "iso_alpha3"]
    ]

    missing_countries = set(
        eu_ember_elec[eu_ember_elec["Country"].isnull()]["iso_alpha3"]
    )
    if any(missing_countries):
        print(
            f"WARNING: Unknown countries in the European Electricity Review data from Ember: {missing_countries}"
        )
    eu_ember_elec = eu_ember_elec.drop(columns="iso_alpha3")

    # Translate the new source groups into the old ones.
    eu_ember_elec["Variable"] = eu_ember_elec["Variable"].replace(
        {
            "Gas": "Gas (TWh)",
            "Hydro": "Hydro (TWh)",
            "Solar": "Solar (TWh)",
            "Wind": "Wind (TWh)",
            "Bioenergy": "Bioenergy (TWh)",
            "Nuclear": "Nuclear (TWh)",
            "Other Fossil": "Oil (TWh)",
            "Other Renewables": "Other renewables excluding bioenergy (TWh)",
        }
    )

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

    # Recalculate total production for European countries.
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

    # Add European Union (27) to population dataset (otherwise the outdated EU 27 data from ember_elec will be used).
    eu_countries = load_eu_27_countries()
    aggregations = {
        col: sum for col in eu_ember_elec.columns if col not in ["Year", "Country"]
    }
    aggregations["Country"] = lambda x: len(list(x))
    eu_ember_elec_added = (
        eu_ember_elec[eu_ember_elec["Country"].isin(eu_countries)]
        .reset_index(drop=True)
        .groupby("Year")
        .agg(aggregations)
        .reset_index()
    )
    # Check that there are indeed 27 countries each year.
    # This is historically inaccurate, but we assume the EU corresponds to its current state.
    assert all(eu_ember_elec_added["Country"] == 27)
    eu_ember_elec_added["Country"] = "European Union (27)"
    eu_ember_elec = pd.concat([eu_ember_elec, eu_ember_elec_added], ignore_index=True)

    return eu_ember_elec


def load_carbon_intensities():
    """Load carbon intensities (in gCO2/kWh) of European countries by combining the latest European Electricity
    Review (which is more up-to-date, but includes only countries in the EU (27)) with an older Ember dataset (which
    includes all EU (27) countries as well as the UK).

    Returns
    -------
    eu_carbon_intensities : pd.DataFrame
        Carbon intensities for european countries.

    """
    # Get EU Ember data on carbon intensities.
    eu_carbon_intensities = read_csv_files_inside_remote_zip(
        url=EMBER_EUROPE_DATA_FILE
    )[EU_EMBER_EMISSIONS_NAME]

    # Select relevant data.
    intensity_col = "Carbon intensity of electricity (gCO2/kWh)"
    columns = {
        "country_name": "Country",
        "year": "Year",
        "emissions_intensity_gco2_kwh": intensity_col,
    }
    eu_carbon_intensities = eu_carbon_intensities.rename(
        errors="raise", columns=columns
    )[columns.values()]

    ####################################################################################################################
    # TODO: Remove this temporary solution once we have a carbon intensity dataset that includes all countries.
    assert "United Kingdom" not in eu_carbon_intensities["Country"].tolist()
    old_carbon_intensities = pd.read_excel(
        EMBER_EUROPE_CARBON_INTENSITY_FILE, sheet_name="Carbon intensities"
    )
    old_carbon_intensities = (
        old_carbon_intensities.melt(
            id_vars=["Area", "Variable"],
            var_name="Year",
            value_name=intensity_col,
        )
        .drop(columns=["Variable"])
        .rename(errors="raise", columns={"Area": "Country"})
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )
    old_carbon_intensities = old_carbon_intensities[
        old_carbon_intensities["Country"] == "United Kingdom"
    ]
    eu_carbon_intensities = pd.concat(
        [eu_carbon_intensities, old_carbon_intensities], ignore_index=True
    )
    ####################################################################################################################

    # Standardize country names.
    eu_carbon_intensities = standardize_countries(
        df=eu_carbon_intensities,
        countries_file=EMBER_EUROPE_COUNTRIES_FILE,
        country_col="Country",
        warn_on_unused_countries=False,
    )

    # Sort conveniently.
    eu_carbon_intensities = eu_carbon_intensities.sort_values(
        ["Country", "Year"]
    ).reset_index(drop=True)

    return eu_carbon_intensities


def load_european_net_imports_and_demand():
    """Load net imports and demand from the European Electricity Review.

    Returns
    -------
    net_imports_and_demand_data : pd.DataFrame
        Net imports and demand data.

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

    net_imports_and_demand_data = standardize_countries(
        df=net_imports_and_demand_data,
        countries_file=EMBER_EUROPE_COUNTRIES_FILE,
        country_col="Country",
    )

    return net_imports_and_demand_data


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


def load_global_ember_data():
    """Load Global electricity data from Ember.

    Returns
    -------
    ember_elec : pd.DataFrame
        Global electricity data from Ember.

    """
    columns = {
        "Area": "Country",
        "Year": "Year",
        "Variable": "Variable",
        "Generation (TWh)": "Value (TWh)",
    }
    # Download global data from Ember.
    response = requests.get(EMBER_GLOBAL_DATA_FILE)
    ember_elec = pd.read_excel(response.content, sheet_name="Data", skiprows=1)
    # Rename and select columns.
    ember_elec = ember_elec.rename(errors="raise", columns=columns)[columns.values()]

    rows = {
        "Production": "Electricity generation (TWh)",
        "Gas": "Gas (TWh)",
        "Coal": "Coal (TWh)",
        "Other fossil": "Oil (TWh)",
        "Nuclear": "Nuclear (TWh)",
        "Hydro": "Hydro (TWh)",
        "Solar": "Solar (TWh)",
        "Wind": "Wind (TWh)",
        "Bioenergy": "Bioenergy (TWh)",
        "Other renewables": "Other renewables excluding bioenergy (TWh)",
        "Demand": "Electricity demand (TWh)",
        "Net imports": "Net imports (TWh)",
    }

    ember_elec["Variable"] = ember_elec["Variable"].replace(rows)

    ember_elec = ember_elec.pivot(
        index=["Country", "Year"], columns="Variable", values="Value (TWh)"
    ).reset_index()[["Country", "Year"] + list(rows.values())]

    # Standardize countries, warn about countries not included in countries file, and remove them.
    ember_elec = standardize_countries(
        df=ember_elec,
        countries_file=EMBER_GLOBAL_COUNTRIES_FILE,
        country_col="Country",
        make_missing_countries_nan=True,
    )
    ember_elec = ember_elec.dropna(subset="Country").reset_index(drop=True)

    return ember_elec


def load_eu_ember_data():
    """Load European electricity data from Ember.

    Returns
    -------
    eu_ember_elec : pd.DataFrame
        European electricity data from Ember.

    """
    # Load updated European Electricity Review data from Ember.
    eu_ember_elec = load_european_electricity_review_data()

    # Load carbon intensities of European countries from older Ember dataset.
    eu_carbon_intensities = load_carbon_intensities()

    # Load data on net imports and demand.
    eu_net_imports_and_demand = load_european_net_imports_and_demand()

    # Add data on carbon intensities to European Electricity Review data.
    eu_ember_elec = pd.merge(
        eu_ember_elec, eu_carbon_intensities, on=["Country", "Year"], how="left"
    )

    # Add data on net imports and demand to European Electricity Review data.
    eu_ember_elec = pd.merge(
        eu_ember_elec, eu_net_imports_and_demand, on=["Country", "Year"], how="left"
    )

    return eu_ember_elec


def combine_global_and_eu_ember_data(global_ember_data, eu_ember_data):
    """Combine Global and European electricity data from Ember.

    Parameters
    ----------
    global_ember_data : pd.DataFrame
        Global electricity data from Ember.
    eu_ember_data : pd.DataFrame
        European electricity data from Ember.

    Returns
    -------
    combined : pd.DataFrame
        Combined data from Ember.

    """
    ember_elec = global_ember_data.copy()
    eu_ember_elec = eu_ember_data.copy()

    # Combine global and European data, and when overlapping, keep the latter (which is more recent).
    combined = pd.concat([ember_elec, eu_ember_elec], ignore_index=True)
    if GLOBAL_VS_EU_EMBER_PRIORITY == "EU":
        combined = combined.drop_duplicates(subset=("Country", "Year"), keep="last")
    else:
        combined = combined.drop_duplicates(subset=("Country", "Year"), keep="first")

    combined = combined.sort_values(["Country", "Year"]).reset_index(drop=True)

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

    print("Import and clean Global Ember data.")
    ember_elec = load_global_ember_data()

    print("Import and clean EU Ember data.")
    eu_ember_elec = load_eu_ember_data()

    print("Combine Global and EU Ember data.")
    ember_data = combine_global_and_eu_ember_data(
        global_ember_data=ember_elec, eu_ember_data=eu_ember_elec
    )

    print("Combine BP and Ember data.")
    combined = combine_bp_and_ember_data(bp_data=bp_data, ember_data=ember_data)

    print("Add per-capita variables.")
    combined = add_per_capita_variables(df=combined)

    print("Add share variables.")
    combined = add_share_variables(df=combined)

    print("Prepare data for grapher.")
    combined = prepare_data_for_grapher(df=combined)

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

    # Save to file as csv
    combined.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    main()
