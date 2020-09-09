import os
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(CURRENT_DIR, "input")
GRAPHER_DIR = os.path.join(CURRENT_DIR, "grapher")

def main():

    # Import total primary energy consumption data from BP
    bp_energy = pd.read_csv(
        os.path.join(INPUT_DIR, "shared/bp_energy.csv"),
        usecols=["Entity", "Year", "Primary Energy Consumption"]
    )
    bp_energy = bp_energy.rename(errors="raise", columns={
        "Entity": "Country",
        "Primary Energy Consumption": "Primary energy consumption (EJ)"
    })

    ej_to_twh = 277.778

    bp_energy["Primary energy consumption (TWh)"] = (
        bp_energy["Primary energy consumption (EJ)"] * ej_to_twh
    )

    bp_energy = bp_energy.drop(errors="raise", columns=["Primary energy consumption (EJ)"])

    bp_energy = bp_energy[-bp_energy["Primary energy consumption (TWh)"].isnull()]

    # Import primary energy consumption data from SHIFT
    shift_consumption = pd.read_csv(
        os.path.join(INPUT_DIR, "energy-consumption/shift_consumption.csv")
    )
    shift_consumption = pd.melt(
        shift_consumption,
        id_vars=["Year"],
        var_name=["Entity"],
        value_name="Primary energy consumption (Mtoe)"
    )

    mtoe_to_twh = 11.63

    shift_consumption["Primary energy consumption (TWh)"] = (
        shift_consumption["Primary energy consumption (Mtoe)"] * mtoe_to_twh
    )

    shift_countries = pd.read_csv(os.path.join(INPUT_DIR, "shared/shift_countries.csv"))
    shift_consumption = (
        shift_consumption
        .merge(shift_countries, on="Entity")
        .drop(errors="raise", columns=["Entity", "Primary energy consumption (Mtoe)"])
        .dropna(subset=["Primary energy consumption (TWh)"])
    )

    # Combine energy consumption from BP and SHIFT
    bp_energy.loc[:, "Source"] = "BP"
    bp_energy.loc[:, "Priority"] = 1

    shift_consumption.loc[:, "Source"] = "SHIFT"
    shift_consumption.loc[:, "Priority"] = 0

    combined = pd.concat([bp_energy, shift_consumption])
    combined = combined.sort_values(["Country", "Year", "Priority"])
    combined = combined.groupby(["Year", "Country"]).tail(1).reset_index(drop=True)

    # Drop columns
    combined = combined.drop(errors="raise", columns=["Priority", "Source"])

    # Calculate annual change
    combined = combined.sort_values(["Country", "Year"])
    combined["Annual change primary energy consumption (%)"] = (
        combined.groupby("Country")["Primary energy consumption (TWh)"].pct_change() * 100
    )
    combined["Annual change primary energy consumption (TWh)"] = (
        combined.groupby("Country")["Primary energy consumption (TWh)"].diff()
    )

    # Calculate per capita energy
    population = pd.read_csv(os.path.join(INPUT_DIR, "shared/population.csv"))

    combined = combined.merge(population,on=["Country","Year"], how="left")
    combined["Energy per capita (kWh)"] = (
        combined["Primary energy consumption (TWh)"] / combined["Population"] * 1000000000
    )
    combined = combined.drop(errors="raise", columns=["Population"])

    # Calculating energy consumption per unit GDP
    gdp = pd.read_csv(
        os.path.join(INPUT_DIR, "shared/total-gdp-maddison.csv"),
        usecols=["Country", "Year", "Total real GDP"]
    )

    combined = combined.merge(gdp, on=["Country", "Year"], how="left")
    combined["Energy per GDP (kWh per $)"] = (
        combined["Primary energy consumption (TWh)"] / combined["Total real GDP"] * 1000000000
    )
    combined = combined.drop(errors="raise", columns=["Total real GDP"])

    # Drop rows with blank Country column
    combined["Country"] = combined["Country"].replace("", np.nan)
    combined = combined.dropna(subset=["Country"])

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)
    combined = combined[combined.isna().sum(axis=1) < len(rounded_cols)]

    # Save to files as csv
    combined.to_csv(
        os.path.join(GRAPHER_DIR, "Primary energy consumption (BP & Shift).csv"),
        index=False
    )

if __name__ == "__main__":
    main()
