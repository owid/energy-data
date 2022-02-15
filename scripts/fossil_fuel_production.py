import os
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(CURRENT_DIR, "input")
GRAPHER_DIR = os.path.join(CURRENT_DIR, "grapher")


def main():

    # Import fossil fuel production data from BP
    bp_fossil = pd.read_csv(
        os.path.join(INPUT_DIR, "shared/bp_energy.csv"),
        usecols=[
            "Entity",
            "Year",
            "Coal Production - EJ",
            "Oil Production - Tonnes",
            "Gas Production - EJ",
        ],
    )

    oil_to_ej = 0.0418

    bp_fossil["Oil Production (EJ)"] = bp_fossil["Oil Production - Tonnes"] * oil_to_ej

    bp_fossil = bp_fossil.rename(
        errors="raise",
        columns={
            "Entity": "Country",
            "Coal Production - EJ": "Coal Production (EJ)",
            "Gas Production - EJ": "Gas Production (EJ)",
        },
    )

    bp_fossil = bp_fossil.drop(errors="raise", columns=["Oil Production - Tonnes"])

    # Import fossil fuel production data from SHIFT
    shift_coal = pd.read_csv(
        os.path.join(INPUT_DIR, "fossil-fuel-production/shift_coal.csv")
    )
    shift_coal = pd.melt(
        shift_coal,
        id_vars=["Year"],
        var_name=["Entity"],
        value_name="Coal Production (EJ)",
    )

    shift_oil = pd.read_csv(
        os.path.join(INPUT_DIR, "fossil-fuel-production/shift_oil.csv")
    )
    shift_oil = pd.melt(
        shift_oil,
        id_vars=["Year"],
        var_name=["Entity"],
        value_name="Oil Production (EJ)",
    )

    shift_gas = pd.read_csv(
        os.path.join(INPUT_DIR, "fossil-fuel-production/shift_gas.csv")
    )
    shift_gas = pd.melt(
        shift_gas,
        id_vars=["Year"],
        var_name=["Entity"],
        value_name="Gas Production (EJ)",
    )

    shift_fossil = shift_coal.merge(shift_oil, on=["Entity", "Year"], how="outer")

    shift_fossil = shift_fossil.merge(shift_gas, on=["Entity", "Year"], how="outer")

    shift_countries = pd.read_csv(os.path.join(INPUT_DIR, "shared/shift_countries.csv"))
    shift_fossil = shift_fossil.merge(shift_countries, on="Entity")
    shift_fossil = shift_fossil.drop(errors="raise", columns=["Entity"])

    # Combine BP and SHIFT data
    bp_fossil.loc[:, "Source"] = "BP"
    bp_fossil.loc[:, "Priority"] = 1

    shift_fossil.loc[:, "Source"] = "SHIFT"
    shift_fossil.loc[:, "Priority"] = 0

    combined = pd.concat([bp_fossil, shift_fossil], join="outer")
    combined = combined.sort_values(["Country", "Year", "Priority"])
    combined = combined.groupby(["Year", "Country"]).tail(1)
    combined = combined.drop(errors="raise", columns=["Priority", "Source"])

    # Convert to TWh
    ej_to_twh = 277.778

    combined["Coal production (TWh)"] = combined["Coal Production (EJ)"] * ej_to_twh
    combined["Oil production (TWh)"] = combined["Oil Production (EJ)"] * ej_to_twh
    combined["Gas production (TWh)"] = combined["Gas Production (EJ)"] * ej_to_twh

    combined = combined.drop(
        errors="raise",
        columns=["Coal Production (EJ)", "Oil Production (EJ)", "Gas Production (EJ)"],
    )

    # Calculate annual change
    combined = combined.sort_values(["Country", "Year"]).reset_index(drop=True)
    for cat in ("Coal", "Oil", "Gas"):
        combined[f"Annual change in {cat.lower()} production (%)"] = (
            combined.groupby("Country")[f"{cat} production (TWh)"].pct_change() * 100
        )
        combined[f"Annual change in {cat.lower()} production (TWh)"] = combined.groupby(
            "Country"
        )[f"{cat} production (TWh)"].diff()

    # Calculate production per capita
    population = pd.read_csv(os.path.join(INPUT_DIR, "shared/population.csv"))

    combined = combined.merge(population, on=["Country", "Year"], how="left")
    for cat in ("Coal", "Oil", "Gas"):
        combined[f"{cat} production per capita (kWh)"] = (
            combined[f"{cat} production (TWh)"] / combined["Population"] * 1000000000
        )
    combined = combined.drop(errors="raise", columns=["Population"])
    combined = combined.replace([np.inf, -np.inf], np.nan)

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(combined) if col not in ("Country", "Year")]
    combined[rounded_cols] = combined[rounded_cols].round(3)
    combined = combined[combined.isna().sum(axis=1) < len(rounded_cols)]

    # Save to files as csv
    combined.to_csv(
        os.path.join(GRAPHER_DIR, "Fossil fuel production (BP & Shift).csv"),
        index=False,
    )


if __name__ == "__main__":
    main()
