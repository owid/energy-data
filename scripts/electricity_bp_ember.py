from functools import reduce
import os
import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(CURRENT_DIR, "input")
GRAPHER_DIR = os.path.join(CURRENT_DIR, "grapher")

def main():

    # Import and clean BP data
    bp_elec = pd.read_csv(os.path.join(INPUT_DIR, "shared/bp_energy.csv"), usecols = [
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
        "Elec Gen from Gas"
    ])

    bp_elec = bp_elec.rename(columns={
        "Primary Energy Consumption": "Primary Energy (Mtoe)",
        "Electricity Generation": "Electricity Generation (TWh)",
        "Entity": "Country",
        "Hydro Generation - TWh": "Hydro (TWh)",
        "Nuclear Generation - TWh": "Nuclear (TWh)",
        "Solar Generation - TWh": "Solar (TWh)",
        "Wind Generation -TWh": "Wind (TWh)",
        "Geo Biomass Other - TWh": "Other renewables (TWh)",
        "Elec Gen from Oil": "Oil (TWh)",
        "Elec Gen from Coal": "Coal (TWh)",
        "Elec Gen from Gas": "Gas (TWh)"
    })

    bp_elec["Fossil fuels (TWh)"] = (
        bp_elec["Coal (TWh)"]
        .add(bp_elec["Oil (TWh)"])
        .add(bp_elec["Gas (TWh)"])
    )
    bp_elec["Renewables (TWh)"] = (
        bp_elec["Hydro (TWh)"]
        .add(bp_elec["Solar (TWh)"])
        .add(bp_elec["Wind (TWh)"])
        .add(bp_elec["Other renewables (TWh)"])
    )
    bp_elec["Low-carbon electricity (TWh)"] = (
        bp_elec["Renewables (TWh)"]
        .add(bp_elec["Nuclear (TWh)"])
    )

    # Convert primary energy to TWh
    mtoe_to_twh = 11.63

    bp_elec["Primary energy (TWh)"] = bp_elec["Primary Energy (Mtoe)"] * mtoe_to_twh
    bp_elec = bp_elec.drop(columns=["Primary Energy (Mtoe)"])

    # Go from wide to long format and drop NAs
    bp_elec = bp_elec.melt(id_vars=["Country", "Year"]).dropna()

    # Import and clean EMBER data
    ember_elec = pd.read_excel(
        os.path.join(INPUT_DIR, "electricity-bp-ember/ember_electricity.xlsx"),
        sheet_name="Data",
        usecols=["Country", "Year", "Variable", "Value (TWh)"]
    )

    sub_dfs = []
    metadata = (
        ("Production", "Electricity Generation (TWh)"),
        ("Gas", "Gas (TWh)"),
        ("Coal", "Coal (TWh)"),
        ("Other fossil", "Oil (TWh)"),
        ("Nuclear", "Nuclear (TWh)"),
        ("Hydro", "Hydro (TWh)"),
        ("Solar", "Solar (TWh)"),
        ("Wind", "Wind (TWh)"),
        ("Biomass and waste", "Biomass and waste"),
        ("Other renewables", "Other renewables"),
    )

    for varname, colname in metadata:
        sub_dfs.append(
            ember_elec[ember_elec["Variable"] == varname]
            .rename(columns={"Value (TWh)": colname})
            .drop(columns=["Variable"])
        )

    ember_elec = reduce(lambda left, right: pd.merge(left, right, on=["Year", "Country"]), sub_dfs)

    ember_countries = pd.read_csv(
        os.path.join(INPUT_DIR, "electricity-bp-ember/ember_countries.csv")
    )

    ember_elec = (
        ember_elec
        .merge(ember_countries, on="Country")
        .drop(columns=["Country"])
        .rename(columns={"OWID Country":"Country"})
    )

    ember_elec["Other renewables (TWh)"] = (
        ember_elec["Other renewables"]
        .add(ember_elec["Biomass and waste"])
    )
    ember_elec["Fossil fuels (TWh)"] = (
        ember_elec["Gas (TWh)"]
        .add(ember_elec["Oil (TWh)"])
        .add(ember_elec["Coal (TWh)"])
    )
    ember_elec["Renewables (TWh)"] = (
        ember_elec["Solar (TWh)"]
        .add(ember_elec["Wind (TWh)"])
        .add(ember_elec["Hydro (TWh)"])
        .add(ember_elec["Other renewables (TWh)"])
    )
    ember_elec["Low-carbon electricity (TWh)"] = (
        ember_elec["Renewables (TWh)"]
        .add(ember_elec["Nuclear (TWh)"])
    )

    ember_elec = ember_elec.drop(columns=["Biomass and waste", "Other renewables"])

    # Reorder columns
    left_columns = ["Country", "Year"]
    other_columns = sorted([col for col in ember_elec.columns if col not in left_columns])
    column_order = left_columns + other_columns
    ember_elec = ember_elec[column_order]

    # Go from wide to long format and drop NAs
    ember_elec = ember_elec.melt(id_vars=["Country", "Year"]).dropna()

    # Combine BP and EMBER
    ember_elec.loc[:, "Source"] = "EMBER"
    ember_elec.loc[:, "Priority"] = 0

    bp_elec.loc[:, "Source"] = "BP"
    bp_elec.loc[:, "Priority"] = 1

    combined = pd.concat([bp_elec, ember_elec])
    combined = combined.sort_values(["Country", "Year", "Priority"])
    combined = combined.groupby(["Year","Country", "variable"]).tail(1)
    combined = combined.drop(columns=["Priority", "Source"])

    # Go back to wide format
    combined = (
        combined
        .pivot_table(values="value", index=["Country", "Year"], columns="variable")
        .reset_index()
    )

    # Calculate per capita electricity
    population = pd.read_csv(os.path.join(INPUT_DIR, "shared/population.csv"))
    combined = combined.merge(population, on=["Country", "Year"])

    for cat in ["Electricity Generation", "Coal", "Oil", "Gas", "Fossil fuels", "Renewables",
                "Low-carbon electricity", "Nuclear", "Solar", "Wind", "Hydro", "Other renewables"]:
        combined[f"{cat} electricity per capita (kWh)"] = (
            combined[f"{cat} (TWh)"] / combined["Population"] * 1000000000
        )

    combined = combined.rename(columns={
        "Electricity Generation electricity per capita (kWh)": "Per capita electricity (kWh)",
        "Low-carbon electricity electricity per capita (kWh)": "Low-carbon electricity per capita (kWh)",
        "Renewables electricity per capita (kWh)": "Renewable electricity per capita (kWh)",
        "Other renewables electricity per capita (kWh)": "Other renewable electricity per capita (kWh)",
        "Fossil fuels electricity per capita (kWh)": "Fossil fuel electricity per capita (kWh)"
    })

    # Drop 'Population' column
    combined = combined.drop(columns=["Population"])

    # Calculate electricity as share of energy
    combined["Electricity as share of primary energy"] = (
        combined["Electricity Generation (TWh)"] / combined["Primary energy (TWh)"] * 100
    )
    combined = combined.drop(columns=["Primary energy (TWh)"])

    # Calculating electricity shares by source
    for cat in ["Coal", "Oil", "Gas", "Fossil fuels", "Renewables", "Low-carbon electricity",
                "Nuclear", "Solar", "Wind", "Hydro", "Other renewables"]:
        combined[f"{cat} (% electricity)"] = (
            combined[f"{cat} (TWh)"] / combined["Electricity Generation (TWh)"] * 100
        )

    # Rename variables for grapher
    combined = combined.rename(columns={
        "Coal (TWh)":"Electricity from coal (TWh)",
        "Gas (TWh)":"Electricity from gas (TWh)",
        "Oil (TWh)":"Electricity from oil (TWh)",
        "Nuclear (TWh)":"Electricity from nuclear (TWh)",
        "Hydro (TWh)":"Electricity from hydro (TWh)",
        "Solar (TWh)":"Electricity from solar (TWh)",
        "Wind (TWh)":"Electricity from wind (TWh)",
        "Other renewables (TWh)":"Electricity from other renewables (TWh)",
        "Fossil fuels (TWh)":"Electricity from fossil fuels (TWh)",
        "Renewables (TWh)":"Electricity from renewables (TWh)"
    })

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)
    combined = combined[combined.isna().sum(axis=1) < len(rounded_cols)]

    # Save to files as csv
    combined.to_csv(
        os.path.join(GRAPHER_DIR, "Electricity mix from BP & EMBER (2020).csv"),
        index=False
    )

if __name__ == "__main__":
    main()
