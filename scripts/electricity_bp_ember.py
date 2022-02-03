"""Combine BP energy and Ember electricity data (global and from Europe).

"""
import os
from functools import reduce

import pandas as pd
from owid import catalog

# Define common paths.
CURRENT_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(CURRENT_DIR, "input")
GRAPHER_DIR = os.path.join(CURRENT_DIR, "grapher")

# Define path to output file of combined dataset.
OUTPUT_FILE = os.path.join(GRAPHER_DIR, "Electricity mix from BP & EMBER (2022).csv")
# Define URL to download Ember 2021 global data from.
EMBER_GLOBAL_DATA_URL = "https://ember-climate.org/wp-content/uploads/2021/03/Data-Global-Electricity-Review-2021.xlsx"
# Define path to Ember 2021 Europe Electricity Review file.
EMBER_EUROPE_FILE = os.path.join(
    INPUT_DIR, "electricity-bp-ember", "EER_2022_generation.csv"
)
# Define path to file with country names in the Ember dataset.
EMBER_COUNTRIES_FILE = os.path.join(
    INPUT_DIR, "electricity-bp-ember", "ember_countries.csv"
)
# Define path to BP energy dataset file.
BP_FILE = os.path.join(INPUT_DIR, "shared", "bp_energy.csv")
#######################################
# TODO: Once owid-catalog is complete, this file will not be necessary.
# Define file with population data.
LEGACY_POPULATION_FILE = os.path.join(INPUT_DIR, "shared", "population.csv")
#######################################
# In case data points are shared by BP and Ember dataset, the source with the highest priority will be taken.
# Assign priority to sources.
EMBER_PRIORITY = 1
BP_PRIORITY = 0

# After analysing the resulting time series, we detected several issues with the following countries/regions.
# For the moment, we remove them from the final dataset.
REGIONS_WITH_INCONSISTENT_DATA = [
    "Moldova",
    "Europe (other)",
    "Other Asia & Pacific",
    "Other CIS",
    "Other Middle East",
]


def prepare_european_electricity_review_data_from_ember():
    """Load and prepare European Electricity Review (EER) electricity data from Ember in a convenient format.

    Generate a dataframe analogous to the one for Ember global electricity, but for European countries, based on the
    European Electricity Review from Ember. Changes to original EER file from Ember:
    * Add 'Hard Coal' to 'Lignite' and call it simply 'Coal'.

    Returns
    -------
    eu_ember_elec : pd.DataFrame
        Ember electricity data for Europe.

    """
    # Upload European Electricity Review 2022 data from Ember.
    eu_ember_elec = pd.read_csv(EMBER_EUROPE_FILE)

    # Change columns for consistency with ember_elec dataframe.
    # I assume that the country code is ISO 3.
    eu_ember_elec = eu_ember_elec[
        ["country_code", "year", "fuel_desc", "generation_twh"]
    ].rename(
        columns={
            "country_code": "iso_alpha3",
            "year": "Year",
            "fuel_desc": "Variable",
            "generation_twh": "Value (TWh)",
        }
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
    ).rename(columns={"name": "Country"})[
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

    return eu_ember_elec


def main():

    # Import and clean BP data
    bp_elec = pd.read_csv(
        BP_FILE,
        usecols=[
            "Entity",
            "Year",
            "Primary Energy Consumption",
            "Electricity Generation ",
            "Hydro Generation - TWh",
            "Nuclear Generation - TWh",
            "Solar Generation - TWh",
            "Wind Generation -TWh",
            "Geo Biomass Other - TWh",
            "Elec Gen from Oil",
            "Elec Gen from Coal",
            "Elec Gen from Gas",
        ],
    )

    bp_elec = bp_elec.rename(
        errors="raise",
        columns={
            "Primary Energy Consumption": "Primary Energy (Mtoe)",
            "Electricity Generation ": "Electricity generation (TWh)",
            "Entity": "Country",
            "Hydro Generation - TWh": "Hydro (TWh)",
            "Nuclear Generation - TWh": "Nuclear (TWh)",
            "Solar Generation - TWh": "Solar (TWh)",
            "Wind Generation -TWh": "Wind (TWh)",
            "Geo Biomass Other - TWh": "Other renewables including bioenergy (TWh)",
            "Elec Gen from Oil": "Oil (TWh)",
            "Elec Gen from Coal": "Coal (TWh)",
            "Elec Gen from Gas": "Gas (TWh)",
        },
    )

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

    # Convert primary energy to TWh
    mtoe_to_twh = 11.63

    bp_elec["Primary energy (TWh)"] = bp_elec["Primary Energy (Mtoe)"] * mtoe_to_twh
    bp_elec = bp_elec.drop(errors="raise", columns=["Primary Energy (Mtoe)"])

    # Go from wide to long format and drop NAs
    bp_elec = bp_elec.melt(id_vars=["Country", "Year"]).dropna()

    # Import and clean Ember data.
    ember_elec = pd.read_excel(EMBER_GLOBAL_DATA_URL, sheet_name="Data", skiprows=1)
    # Rename columns and variables for consistency with previous version of the dataset.
    ember_elec = ember_elec.rename(
        columns={"Area": "Country", "Generation (TWh)": "Value (TWh)"}
    )[["Country", "Year", "Variable", "Value (TWh)"]]

    sub_dfs = []
    metadata = (
        ("Production", "Electricity generation (TWh)"),
        ("Gas", "Gas (TWh)"),
        ("Coal", "Coal (TWh)"),
        ("Other fossil", "Oil (TWh)"),
        ("Nuclear", "Nuclear (TWh)"),
        ("Hydro", "Hydro (TWh)"),
        ("Solar", "Solar (TWh)"),
        ("Wind", "Wind (TWh)"),
        ("Bioenergy", "Bioenergy (TWh)"),
        ("Other renewables", "Other renewables excluding bioenergy (TWh)"),
    )

    for varname, colname in metadata:
        sub_dfs.append(
            ember_elec[ember_elec["Variable"] == varname]
            .rename(errors="raise", columns={"Value (TWh)": colname})
            .drop(errors="raise", columns=["Variable"])
        )

    ember_elec = reduce(
        lambda left, right: pd.merge(left, right, on=["Year", "Country"]), sub_dfs
    )

    ember_countries = pd.read_csv(EMBER_COUNTRIES_FILE)

    ember_elec = (
        ember_elec.merge(ember_countries, on="Country")
        .drop(errors="raise", columns=["Country"])
        .rename(errors="raise", columns={"OWID Country": "Country"})
    )

    # Load updated European Electricity Review data from Ember.
    eu_ember_elec = prepare_european_electricity_review_data_from_ember()

    # Replace only those rows where country & year correspond to those in the new file.
    ember_elec = (
        pd.concat([ember_elec, eu_ember_elec], ignore_index=True)
        .drop_duplicates(subset=("Country", "Year"), keep="last")
        .sort_values(["Country", "Year"])
    )

    ember_elec["Other renewables including bioenergy (TWh)"] = ember_elec[
        "Other renewables excluding bioenergy (TWh)"
    ].add(ember_elec["Bioenergy (TWh)"])
    ember_elec["Fossil fuels (TWh)"] = (
        ember_elec["Gas (TWh)"]
        .add(ember_elec["Oil (TWh)"])
        .add(ember_elec["Coal (TWh)"])
    )
    ember_elec["Renewables (TWh)"] = (
        ember_elec["Solar (TWh)"]
        .add(ember_elec["Wind (TWh)"])
        .add(ember_elec["Hydro (TWh)"])
        .add(ember_elec["Bioenergy (TWh)"])
        .add(ember_elec["Other renewables excluding bioenergy (TWh)"])
    )
    ember_elec["Low-carbon electricity (TWh)"] = ember_elec["Renewables (TWh)"].add(
        ember_elec["Nuclear (TWh)"]
    )

    # Reorder columns
    left_columns = ["Country", "Year"]
    other_columns = sorted(
        [col for col in ember_elec.columns if col not in left_columns]
    )
    column_order = left_columns + other_columns
    ember_elec = ember_elec[column_order]

    # Go from wide to long format and drop NAs
    ember_elec = ember_elec.melt(id_vars=["Country", "Year"]).dropna()

    # Combine BP and EMBER
    ember_elec.loc[:, "Source"] = "EMBER"
    ember_elec.loc[:, "Priority"] = EMBER_PRIORITY

    bp_elec.loc[:, "Source"] = "BP"
    bp_elec.loc[:, "Priority"] = BP_PRIORITY

    combined = pd.concat([bp_elec, ember_elec])
    combined = combined.sort_values(["Country", "Year", "Priority"])
    combined = combined.groupby(["Year", "Country", "variable"]).tail(1)
    combined = combined.drop(errors="raise", columns=["Priority", "Source"])

    # Go back to wide format
    combined = combined.pivot_table(
        values="value", index=["Country", "Year"], columns="variable"
    ).reset_index()

    # Calculate per capita electricity
    population = (
        catalog.find("population", namespace="owid")
        .load()
        .reset_index()
        .rename(
            columns={"country": "Country", "year": "Year", "population": "Population"}
        )[["Country", "Year", "Population"]]
    )
    ##################################################
    # TODO: Remove this temporary solution once all countries and regions have been added to owid-catalog.
    additional_population = pd.read_csv(LEGACY_POPULATION_FILE)
    population = (
        pd.concat([population, additional_population], ignore_index=True)
        .drop_duplicates(subset=["Country", "Year"], keep="first")
        .sort_values(["Country", "Year"])
    )
    ##################################################

    combined = combined.merge(population, on=["Country", "Year"], how="left")

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
            combined[f"{cat} (TWh)"] / combined["Population"] * 1e9
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

    # Calculate electricity as share of energy
    combined["Electricity as share of primary energy"] = (
        combined["Electricity generation (TWh)"]
        / combined["Primary energy (TWh)"]
        * 100
    )
    combined = combined.drop(errors="raise", columns=["Primary energy (TWh)"])

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

    #########################################
    # TODO: Remove this temporary solution once inconsistencies in data have been tackled.
    #  For the moment, remove countries and regions with inconsistent data.
    print(f"WARNING: Removing countries and regions with inconsistent data:")
    for region in REGIONS_WITH_INCONSISTENT_DATA:
        print(f" * {region}")
    combined = combined[
        ~combined["Country"].isin(REGIONS_WITH_INCONSISTENT_DATA)
    ].reset_index(drop=True)
    #########################################

    # Save to file as csv
    combined.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
