import os
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(CURRENT_DIR, "input")
GRAPHER_DIR = os.path.join(CURRENT_DIR, "grapher")

def main():

    primary_energy = pd.read_csv(os.path.join(INPUT_DIR, "shared/bp_energy.csv"), usecols=[
        "Entity",
        "Year",
        "Coal Consumption - EJ",
        "Gas Consumption - EJ",
        "Oil Consumption - EJ",
        "Hydro Consumption - EJ",
        "Nuclear Consumption - EJ",
        "Biofuels Consumption - PJ - Total",
        "Primary Energy Consumption",
        "Solar Consumption - EJ",
        "Wind Consumption - EJ",
        "Geo Biomass Other - EJ",
        "Hydro Generation - TWh",
        "Nuclear Generation - TWh",
        "Solar Generation - TWh",
        "Wind Generation -TWh",
        "Geo Biomass Other - TWh"
    ])

    primary_energy = primary_energy.rename(columns={
        "Coal Consumption - EJ": "Coal (EJ)",
        "Gas Consumption - EJ": "Gas (EJ)",
        "Oil Consumption - EJ": "Oil (EJ)",
        "Hydro Consumption - EJ": "Hydro (EJ)",
        "Nuclear Consumption - EJ": "Nuclear (EJ)",
        "Solar Consumption - EJ": "Solar (EJ)",
        "Wind Consumption - EJ": "Wind (EJ)",
        "Geo Biomass Other - EJ": "Other renewables (EJ)",
        "Primary Energy Consumption": "Primary Energy (EJ)",
        "Entity": "Country",
        "Hydro Generation - TWh": "Hydro (TWh)",
        "Nuclear Generation - TWh": "Nuclear (TWh)",
        "Solar Generation - TWh": "Solar (TWh)",
        "Wind Generation -TWh": "Wind (TWh)",
        "Geo Biomass Other - TWh": "Other renewables (TWh)",
        "Biofuels Consumption - PJ - Total": "Biofuels (PJ)"
    })

    primary_energy["Biofuels (PJ)"] = primary_energy["Biofuels (PJ)"].fillna(0)

    pj_to_ej = 0.001

    primary_energy["Biofuels (EJ)"] = primary_energy["Biofuels (PJ)"] * pj_to_ej
    primary_energy["Fossil Fuels (EJ)"] = (
        primary_energy["Coal (EJ)"]
        .add(primary_energy["Oil (EJ)"])
        .add(primary_energy["Gas (EJ)"])
    )
    primary_energy["Renewables (EJ)"] = (
        primary_energy["Hydro (EJ)"]
        .add(primary_energy["Solar (EJ)"])
        .add(primary_energy["Wind (EJ)"])
        .add(primary_energy["Other renewables (EJ)"])
        .add(primary_energy["Biofuels (EJ)"])
    )
    primary_energy["Low-carbon energy (EJ)"] = (
        primary_energy["Renewables (EJ)"]
        .add(primary_energy["Nuclear (EJ)"])
    )

    # Converting all sources to TWh (primary energy – sub method)
    ej_to_twh = 277.778

    for cat in ["Coal", "Oil", "Gas", "Biofuels"]:
        primary_energy[f"{cat} (TWh)"] = primary_energy[f"{cat} (EJ)"] * ej_to_twh

    for cat in ["Hydro", "Nuclear", "Renewables", "Solar", "Wind",
                "Other renewables", "Low-carbon energy"]:
        primary_energy[f"{cat} (TWh – sub method)"] = primary_energy[f"{cat} (EJ)"] * ej_to_twh

    primary_energy["Renewables (TWh)"] = (
        primary_energy["Hydro (TWh)"]
        .add(primary_energy["Solar (TWh)"])
        .add(primary_energy["Wind (TWh)"])
        .add(primary_energy["Other renewables (TWh)"])
        .add(primary_energy["Biofuels (TWh)"])
    )
    primary_energy["Low-carbon energy (TWh)"] = (
        primary_energy["Renewables (TWh)"]
        .add(primary_energy["Nuclear (TWh)"])
    )
    primary_energy["Fossil Fuels (TWh)"] = (
        primary_energy["Coal (TWh)"]
        .add(primary_energy["Oil (TWh)"])
        .add(primary_energy["Gas (TWh)"])
    )
    primary_energy["Primary energy (TWh)"] = (
        primary_energy["Fossil Fuels (TWh)"]
        .add(primary_energy["Low-carbon energy (TWh – sub method)"])
    )

    # Calculating each source as share of direct primary energy
    primary_energy["Primary energy – direct (TWh)"] = (
        primary_energy["Fossil Fuels (TWh)"] + primary_energy["Low-carbon energy (TWh)"]
    )

    for cat in ["Coal", "Gas", "Oil", "Biofuels", "Nuclear", "Hydro", "Renewables",
                "Solar", "Wind", "Other renewables", "Fossil Fuels", "Low-carbon energy"]:
        primary_energy[f"{cat} (% primary direct energy)"] = (
            primary_energy[f"{cat} (TWh)"] / primary_energy["Primary energy – direct (TWh)"] * 100
        )

    # Calculating each source as share of energy (substitution method)
    for cat in ["Coal", "Gas", "Oil", "Biofuels", "Nuclear", "Hydro", "Renewables",
                "Solar", "Wind", "Other renewables", "Fossil Fuels", "Low-carbon energy"]:
        primary_energy[f"{cat} (% sub energy)"] = (
            primary_energy[f"{cat} (EJ)"] / primary_energy["Primary Energy (EJ)"] * 100
        )

    # Calculating annual change in each source
    primary_energy = primary_energy.sort_values(["Country","Year"])

    for cat in ["Coal", "Oil", "Gas", "Biofuels", "Fossil Fuels"]:
        primary_energy[f"{cat} (% growth)"] = (
            primary_energy.groupby("Country")[f"{cat} (TWh)"].pct_change() * 100
        )
        primary_energy[f"{cat} (TWh growth – sub method)"] = (
            primary_energy.groupby("Country")[f"{cat} (TWh)"].diff()
        )

    for cat in ["Hydro", "Nuclear", "Renewables", "Solar", "Wind",
                "Other renewables", "Low-carbon energy"]:
        primary_energy[f"{cat} (% growth)"] = (
            primary_energy.groupby("Country")[f"{cat} (TWh – sub method)"].pct_change() * 100
        )
        primary_energy[f"{cat} (TWh growth – sub method)"] = (
            primary_energy.groupby("Country")[f"{cat} (TWh – sub method)"].diff()
        )

    # Calculate per capita energy
    population = pd.read_csv(os.path.join(INPUT_DIR, "shared/population.csv"))

    primary_energy = primary_energy.merge(population, on=["Country", "Year"])

    for cat in ["Coal", "Oil", "Gas", "Biofuels", "Fossil Fuels"]:
        primary_energy[f"{cat} per capita (kWh)"] = (
            primary_energy[f"{cat} (TWh)"] / primary_energy["Population"] * 1000000000
        )

    for cat in ["Hydro", "Nuclear", "Renewables", "Solar", "Wind",
                "Other renewables", "Low-carbon energy"]:
        primary_energy[f"{cat} per capita (kWh)"] = (
            primary_energy[f"{cat} (TWh – sub method)"] / primary_energy["Population"] * 1000000000
        )

    energy_mix = primary_energy.drop(columns=[
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
        "Population"
    ])

    energy_mix = energy_mix.replace([np.inf, -np.inf], np.nan)

    # Round all values to 3 decimal places
    rounded_cols = [col for col in list(energy_mix) if col not in ("Country", "Year")]
    energy_mix[rounded_cols] = energy_mix[rounded_cols].round(3)

    # Save to files as csv
    energy_mix.to_csv(
        os.path.join(GRAPHER_DIR, "Energy mix from BP (2020).csv"),
        index=False
    )

if __name__ == "__main__":
    main()
