"""Functions shared among different modules.

TODO: Consider moving some (or all) of these functions to data-utils.

"""

import json

import numpy as np
import pandas as pd

from owid import catalog


class ExceptionFromDocstring(Exception):
    """Exception that, if no exception message is explicitly given, returns its own docstring."""

    def __init__(self, exception_message=None, *args):
        super().__init__(exception_message or self.__doc__, *args)


class DataFramesHaveDifferentLengths(ExceptionFromDocstring):
    """Dataframes cannot be compared because they have different number of rows."""


class ObjectsAreNotDataframes(ExceptionFromDocstring):
    """Given objects are not dataframes."""


def compare_dataframes(
    df1, df2, columns=None, absolute_tolerance=1e-8, relative_tolerance=1e-8
):
    """Compare two dataframes element by element, assuming that nans are all identical, and assuming certain absolute
    and relative tolerances for the comparison of floats.

    NOTE: Dataframes must have the same number of rows to be able to compare them.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe.
    df2 : pd.DataFrame
        Second dataframe.
    columns : list or None
        List of columns to compare (they both must exist in both dataframes). If None, common columns will be compared.
    absolute_tolerance : float
        Absolute tolerance to assume in the comparison of each cell in the dataframes. A value a of an element in df1 is
        considered equal to the corresponding element b at the same position in df2, if:
        abs(a - b) <= absolute_tolerance
    relative_tolerance : float
        Relative tolerance to assume in the comparison of each cell in the dataframes. A value a of an element in df1 is
        considered equal to the corresponding element b at the same position in df2, if:
        abs(a - b) / abs(b) <= relative_tolerance

    Returns
    -------
    compared : pd.DataFrame
        Dataframe with as many rows as df1 and df2, and as many columns as specified by `columns` argument (or as many
        common columns between df1 and df2, if `columns` is None).

    """
    # Ensure dataframes can be compared.
    if (type(df1) != pd.DataFrame) or (type(df2) != pd.DataFrame):
        raise ObjectsAreNotDataframes
    if len(df1) != len(df2):
        raise DataFramesHaveDifferentLengths

    # If columns are not specified, assume common columns.
    if columns is None:
        columns = sorted(set(df1.columns) & set(df2.columns))

    # Compare, column by column, the elements of the two dataframes.
    compared = pd.DataFrame()
    for col in columns:
        if (df1[col].dtype == object) or (df2[col].dtype == object):
            # Apply a direct comparison for strings.
            compared_row = df1[col].values == df2[col].values
        else:
            # For numeric data, consider them equal within certain absolute and relative tolerances.
            compared_row = np.isclose(
                df1[col].values,
                df2[col].values,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
            # Treat nans as equal.
            compared_row[np.isnan(df1[col].values) & np.isnan(df2[col].values)] = True
        compared[col] = compared_row

    return compared


def are_dataframes_equal(df1, df2, absolute_tolerance=1e-8, relative_tolerance=1e-8):
    """Check whether two dataframes are equal, assuming that all nans are identical, and comparing floats by means of
    certain absolute and relative tolerances.

    Parameters
    ----------
    df1 : pd.DataFrame
        First dataframe.
    df2 : pd.DataFrame
        Second dataframe.
    absolute_tolerance : float
        Absolute tolerance to assume in the comparison of each cell in the dataframes. A value a of an element in df1 is
        considered equal to the corresponding element b at the same position in df2, if:
        abs(a - b) <= absolute_tolerance
    relative_tolerance : float
        Relative tolerance to assume in the comparison of each cell in the dataframes. A value a of an element in df1 is
        considered equal to the corresponding element b at the same position in df2, if:
        abs(a - b) / abs(b) <= relative_tolerance

    Returns
    -------
    are_equal : bool
        True if the two dataframes are equal (given the conditions explained above).
    compared : pd.DataFrame
        Dataframe with the same shape as df1 and df2 (if they have the same shape) that is True on each element where
        both dataframes have equal values. If dataframes have different shapes, compared will be empty.

    """
    # Initialise flag that is True only if both dataframes are equal.
    are_equal = True
    # Initialise flag that is True if dataframes can be compared cell by cell.
    can_be_compared = True

    # Check if all columns in df2 are in df1.
    missing_in_df1 = sorted(set(df2.columns) - set(df1.columns))
    if len(missing_in_df1):
        print(f"* {len(missing_in_df1)} columns in df2 missing in df1.")
        print("\n".join([f"  * {col}" for col in missing_in_df1]))
        are_equal = False

    # Check if all columns in df1 are in df2.
    missing_in_df2 = sorted(set(df1.columns) - set(df2.columns))
    if len(missing_in_df2):
        print(f"* {len(missing_in_df2)} columns in df1 missing in df2.")
        print("\n".join([f"  * {col}" for col in missing_in_df2]))
        are_equal = False

    # Check if dataframes have the same number of rows.
    if len(df1) != len(df2):
        print(f"* {len(df1)} rows in df1 and {len(df2)} rows in df2.")
        are_equal = False
        can_be_compared = False

    # Check for differences in column names or types.
    common_columns = sorted(set(df1.columns) & set(df2.columns))
    all_columns = sorted(set(df1.columns) | set(df2.columns))
    if common_columns == all_columns:
        if df1.columns.tolist() != df2.columns.tolist():
            print("* Columns are sorted differently.")
            are_equal = False
        for col in common_columns:
            if df1[col].dtype != df2[col].dtype:
                print(
                    f"  * Column {col} is of type {df1[col].dtype} for df1, but type {df2[col].dtype} for df2."
                )
                are_equal = False
    else:
        print(
            f"* Only {len(common_columns)} common columns out of {len(all_columns)} distinct columns."
        )
        are_equal = False

    if not can_be_compared:
        # Dataframes cannot be compared.
        compared = pd.DataFrame()
        are_equal = False
    else:
        # Check if indexes are equal.
        if (df1.index != df2.index).any():
            print(
                "* Dataframes have different indexes (consider resetting indexes of input dataframes)."
            )
            are_equal = False

        # Dataframes can be compared cell by cell.
        compared = compare_dataframes(
            df1,
            df2,
            columns=common_columns,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        # Dataframes are equal only if all previous checks have passed and cells are identical
        # Two nans are considered identical.
        are_equal = are_equal & compared.all().all()

    if are_equal:
        print(
            f"Dataframes are identical (within absolute tolerance of {absolute_tolerance} and relative tolerance of "
            f"{relative_tolerance})."
        )

    return are_equal, compared


def _warn_on_list_of_entities(list_of_entities, warning_message, show_list):
    print(warning_message)
    if show_list:
        print("\n".join(["* " + entity for entity in list_of_entities]))


def standardize_countries(
    df,
    countries_file,
    country_col="country",
    warn_on_missing_countries=True,
    make_missing_countries_nan=False,
    warn_on_unused_countries=True,
    show_full_warning=True,
):
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
                show_list=show_full_warning,
            )
        if make_missing_countries_nan:
            df_standardized.loc[
                df_standardized[country_col].isin(missing_countries), country_col
            ] = np.nan

    # Optionally warn if countries file has entities that have not been found in dataframe.
    if warn_on_unused_countries:
        unused_countries = sorted(set(countries) - set(df[country_col]))
        if len(unused_countries) > 0:
            _warn_on_list_of_entities(
                list_of_entities=unused_countries,
                warning_message=f"WARNING: {len(unused_countries)} unused entities in countries file.",
                show_list=show_full_warning,
            )

    return df_standardized


def add_population_to_dataframe(
    df,
    country_col="country",
    year_col="year",
    population_col="population",
    warn_on_missing_countries=True,
    show_full_warning=True,
):
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
    population = (
        catalog.find("population", namespace="owid", dataset="key_indicators")
        .load()
        .reset_index()
        .rename(
            columns={
                "country": country_col,
                "year": year_col,
                "population": population_col,
            }
        )[[country_col, year_col, population_col]]
    )

    # Check if there is any missing country.
    missing_countries = set(df[country_col]) - set(population[country_col])
    if len(missing_countries) > 0:
        if warn_on_missing_countries:
            _warn_on_list_of_entities(
                list_of_entities=missing_countries,
                warning_message=f"WARNING: {len(missing_countries)} countries not found in population dataset. "
                f"They will remain in the dataset, but have nan population.",
                show_list=show_full_warning,
            )

    # Add population to original dataframe.
    df_with_population = pd.merge(
        df, population, on=[country_col, year_col], how="left"
    )

    return df_with_population
