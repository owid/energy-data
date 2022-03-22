"""Functions shared among different modules.

"""

import json

import numpy as np
import pandas as pd

from owid import catalog


def _warn_on_list_of_entities(list_of_entities, warning_message, show_list):
    print(warning_message)
    if show_list:
        print("\n".join(["* " + entity for entity in list_of_entities]))


def standardize_countries(df, countries_file, country_col='country', warn_on_missing_countries=True,
                          make_missing_countries_nan=False, warn_on_unused_countries=True, show_full_warning=True):
    """Standardize country names in dataframe, following the mapping given in a file.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe that contains a column of non-standardized country names.
    countries_file : str
        Path to json file containing a mapping from non-standardized to standardized country names.
    country_col : str
        Name of column in df containing non-standardized country names.
    warn_on_missing_countries : bool
        True to warn about countries that appear in original table but not in countries file.
    make_missing_countries_nan : bool
        True to make nan any country that appears in original dataframe but not in countries file. False to keep their
        original (possibly non-standardized) names.
    warn_on_unused_countries : bool
        True to warn about countries that appear in countries file but are useless (since they do not appear in original
        dataframe).
    show_full_warning : bool
        True to display list of countries in warning messages.

    Returns
    -------
    df_standardized : pd.DataFrame
        Original dataframe after standardizing the column of country names.

    """
    # Load country mappings.
    with open(countries_file, "r") as _countries:
        countries = json.loads(_countries.read())

    # Find countries that exist in dataframe but are missing in (left column of) countries file.
    missing_countries = sorted(set(df[country_col]) - set(countries))

    # Replace country names following the mapping given in the countries file.
    # Countries in dataframe that are not among countries, will be left unchanged.
    df_standardized = df.copy()
    df_standardized[country_col] = df[country_col].replace(countries)

    # Decide what to do with missing countries.
    if len(missing_countries) > 0:
        if warn_on_missing_countries:
            _warn_on_list_of_entities(
                list_of_entities=missing_countries,
                warning_message=f"WARNING: {len(missing_countries)} entities in dataframe missing in countries file.",
                show_list=show_full_warning)
        if make_missing_countries_nan:
            df_standardized.loc[df_standardized[country_col].isin(missing_countries), country_col] = np.nan

    # Optionally warn if countries file has entities that have not been found in dataframe.
    if warn_on_unused_countries:
        unused_countries = sorted(set(countries) - set(df[country_col]))
        if len(unused_countries) > 0:
            _warn_on_list_of_entities(
                list_of_entities=unused_countries,
                warning_message=f"WARNING: {len(unused_countries)} unused entities in countries file.",
                show_list=show_full_warning)

    return df_standardized


def add_population_to_dataframe(df, country_col='country', year_col='year', population_col='population',
                                warn_on_missing_countries=True, show_full_warning=True):
    """Add column of population to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe that contains a column of country names and years.
    country_col : str
        Name of column in original dataframe with country names.
    year_col : str
        Name of column in original dataframe with years.
    population_col : str
        Name of new column to be created with population values.
    warn_on_missing_countries : bool
        True to warn about countries that appear in original dataframe but not in population dataset.
    show_full_warning : bool
        True to display list of countries in warning messages.

    Returns
    -------
    df_with_population : pd.DataFrame
        Original dataframe after adding a column with population values.

    """
    # Load population data and calculate per capita energy.
    population = catalog.find("population", namespace="owid", dataset="key_indicators").load().reset_index().rename(
            columns={"country": country_col, "year": year_col, "population": population_col}
    )[[country_col, year_col, population_col]]

    # Check if there is any missing country.
    missing_countries = set(df[country_col]) - set(population[country_col])
    if len(missing_countries) > 0:
        if warn_on_missing_countries:
            _warn_on_list_of_entities(
                list_of_entities=missing_countries,
                warning_message=f"WARNING: {len(missing_countries)} countries not found in population dataset. "
                                f"They will remain in the dataset, but have nan population.",
                show_list=show_full_warning)

    # Add population to original dataframe.
    df_with_population = pd.merge(df, population, on=[country_col, year_col], how='left')

    return df_with_population
