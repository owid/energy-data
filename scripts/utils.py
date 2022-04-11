"""Functions shared among different modules.

TODO: Consider moving some (or all) of these functions to data-utils.

"""

import json
import warnings

import numpy as np
import pandas as pd

from owid import catalog

# Default parameters for aggregation of data to construct regions.
MIN_FRAC_INDIVIDUAL_POPULATION = 0.0
MIN_FRAC_CUMULATIVE_POPULATION = 0.7
REFERENCE_YEAR = 2018
FRAC_ALLOWED_NANS_PER_YEAR = 0.2
NUM_ALLOWED_NANS_PER_YEAR = None


class ExceptionFromDocstring(Exception):
    """Exception that, if no exception message is explicitly given, returns its own docstring."""

    def __init__(self, exception_message=None, *args):
        super().__init__(exception_message or self.__doc__, *args)


class DataFramesHaveDifferentLengths(ExceptionFromDocstring):
    """Dataframes cannot be compared because they have different number of rows."""


class ObjectsAreNotDataframes(ExceptionFromDocstring):
    """Given objects are not dataframes."""


def _load_population():
    population = (
        catalog.find("population", namespace="owid", dataset="key_indicators")
        .load()
        .reset_index()
    )

    return population


def _load_countries_regions():
    countries_regions = catalog.find(
        "countries_regions", dataset="reference", namespace="owid"
    ).load()

    return countries_regions


def _load_income_groups():
    income_groups = catalog.find(
        table="wb_income_group", dataset="wb_income", namespace="wb"
    )
    income_groups = (
        income_groups[income_groups.path.str.startswith("garden")].load().reset_index()
    )

    return income_groups


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
        compared_row[pd.isnull(df1[col].values) & pd.isnull(df2[col].values)] = True
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
    population = _load_population().rename(
        columns={
            "country": country_col,
            "year": year_col,
            "population": population_col,
        }
    )[[country_col, year_col, population_col]]

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


def groupby_agg(
    df, groupby_columns, aggregations=None, num_allowed_nans=0, frac_allowed_nans=None
):
    """Group dataframe by certain columns, and aggregate using a certain method, and decide how to handle nans.

    This function is similar to the usual
    > df.groupby(groupby_columns).agg(aggregations)
    However, pandas by default ignores nans in aggregations. This implies, for example, that
    > df.groupby(groupby_columns).sum()
    will treat nans as zeros, which can be misleading.

    When both num_allowed_nans and frac_allowed_nans are None, this function behaves like the default pandas behaviour
    (and nans will be treated as zeros).

    On the other hand, if num_allowed_nans is not None, then a group will be nan if the number of nans in that group is
    larger than num_allowed_nans, otherwise nans will be treated as zeros.

    Similarly, if frac_allowed_nans is not None, then a group will be nan if the fraction of nans in that group is
    larger than frac_allowed_nans, otherwise nans will be treated as zeros.

    If both num_allowed_nans and frac_allowed_nans are not None, both conditions are applied. This means that, each
    group must have a number of nans <= num_allowed_nans, and a fraction of nans <= frac_allowed_nans, otherwise that
    group will be nan.

    Note: This function won't work when using multiple aggregations for the same column (e.g. {'a': ('sum', 'mean')}).

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe.
    groupby_columns : list or str
        List of columns to group by. It can be given as a string, if it is only one column.
    aggregations : dict or None
        Aggregations to apply to each column in df. If None, 'sum' will be applied to all columns.
    num_allowed_nans : int or None
        Maximum number of nans that are allowed in a group.
    frac_allowed_nans : float or None
        Maximum fraction of nans that are allowed in a group.

    Returns
    -------
    grouped : pd.DataFrame
        Grouped dataframe after applying aggregations.

    """
    if type(groupby_columns) == str:
        groupby_columns = [groupby_columns]

    if aggregations is None:
        columns_to_aggregate = [
            column for column in df.columns if column not in groupby_columns
        ]
        aggregations = {column: "sum" for column in columns_to_aggregate}

    # Group by and aggregate.
    grouped = df.groupby(groupby_columns, dropna=False).agg(aggregations)

    if num_allowed_nans is not None:
        # Count the number of missing values in each group.
        num_nans_detected = df.groupby(groupby_columns, dropna=False).agg(
            lambda x: pd.isnull(x).sum()
        )
        # Make nan any aggregation where there were too many missing values.
        grouped = grouped[num_nans_detected <= num_allowed_nans]

    if frac_allowed_nans is not None:
        # Count the number of missing values in each group.
        num_nans_detected = df.groupby(groupby_columns, dropna=False).agg(
            lambda x: pd.isnull(x).sum()
        )
        # Count number of elements in each group (avoid using 'count' method, which ignores nans).
        num_elements = df.groupby(groupby_columns, dropna=False).size()
        # Make nan any aggregation where there were too many missing values.
        grouped = grouped[
            num_nans_detected.divide(num_elements, axis="index") <= frac_allowed_nans
        ]

    return grouped


class RegionNotFound(ExceptionFromDocstring):
    """Region was not found in countries-regions dataset."""


def list_countries_in_region(region, countries_regions=None, income_groups=None):
    """List countries that are members of a region.

    Parameters
    ----------
    region : str
        Name of the region (e.g. Europe).
    countries_regions : pd.DataFrame or None
        Countries-regions dataset, or None to load it from the catalog.
    income_groups : pd.DataFrame or None
        Income-groups dataset, or None, to load it from the catalog.

    Returns
    -------
    members : list
        Names of countries that are members of the region.

    """
    if countries_regions is None:
        countries_regions = _load_countries_regions()

    # TODO: Remove lines related to income_groups once they are included in countries-regions dataset.
    if income_groups is None:
        income_groups = _load_income_groups()
    income_groups_names = income_groups["income_group"].dropna().unique().tolist()

    # TODO: Once countries-regions has additional columns 'is_historic' and 'is_country', select only countries, and not
    #  historical regions.
    if region in countries_regions["name"].tolist():
        # Find codes of member countries in this region.
        member_codes_str = countries_regions[countries_regions["name"] == region][
            "members"
        ].item()
        if pd.isnull(member_codes_str):
            member_codes = []
        else:
            member_codes = json.loads(member_codes_str)
        # Get standardized names of these countries.
        members = countries_regions.loc[member_codes]["name"].tolist()
    elif region in income_groups_names:
        members = (
            income_groups[income_groups["income_group"] == region]["country"]
            .unique()
            .tolist()
        )
    else:
        raise RegionNotFound

    return members


def list_countries_in_region_that_must_have_data(
    region,
    reference_year=REFERENCE_YEAR,
    min_frac_individual_population=MIN_FRAC_INDIVIDUAL_POPULATION,
    min_frac_cumulative_population=MIN_FRAC_CUMULATIVE_POPULATION,
    countries_regions=None,
    income_groups=None,
    population=None,
):
    """List countries of a region that are expected to have the largest contribution to any variable (based on their
    population).

    Method to select countries:
    1. Select countries whose population is, on a certain reference year (reference_year), larger than a fraction of
      min_frac_individual_population with respect to the total population of the region.
    2. Among those, sort countries by descending population, and cut as soon as the cumulative population exceeds
      min_frac_cumulative_population.
    Note: It may not be possible to fulfil both conditions. In that case, a warning is raised.

    Parameters
    ----------
    region : str
        Name of the region.
    reference_year : int
        Reference year to consider when selecting countries.
    min_frac_individual_population : float
        Minimum fraction of the total population of the region that each of the listed countries must exceed.
    min_frac_cumulative_population : float
        Minimum fraction of the total population of the region that the sum of the listed countries must exceed.
    countries_regions : pd.DataFrame or None
        Countries-regions dataset, or None, to load it from owid catalog.
    income_groups : pd.DataFrame or None
        Income-groups dataset, or None, to load it from the catalog.
    population : pd.DataFrame or None
        Population dataset, or None, to load it from owid catalog.

    Returns
    -------
    countries : list
        List of countries that are expected to have the largest contribution.

    """
    if countries_regions is None:
        countries_regions = _load_countries_regions()

    if population is None:
        population = _load_population()

    if income_groups is None:
        income_groups = _load_income_groups()

    # List all countries in the selected region.
    members = list_countries_in_region(
        region, countries_regions=countries_regions, income_groups=income_groups
    )

    # Select population data for reference year for all countries in the region.
    reference = (
        population[
            (population["country"].isin(members))
            & (population["year"] == reference_year)
        ]
        .dropna(subset="population")
        .sort_values("population", ascending=False)
        .reset_index(drop=True)
    )

    # Calculate total population in the region, and the fractional contribution of each country.
    total_population = reference["population"].sum()
    reference["fraction"] = reference["population"] / total_population

    # Select countries that exceed a minimum individual fraction of the total population of the region.
    selected = reference[
        (reference["fraction"] > min_frac_individual_population)
    ].reset_index(drop=True)

    # Among remaining countries, select countries that, combined, exceed a minimum fraction of the total population.
    selected["cumulative_fraction"] = selected["population"].cumsum() / total_population
    candidates_to_ignore = selected[
        selected["cumulative_fraction"] > min_frac_cumulative_population
    ]
    if len(candidates_to_ignore) > 0:
        selected = selected.loc[0 : candidates_to_ignore.index[0]]

    if (min_frac_individual_population == 0) and (min_frac_cumulative_population == 0):
        warnings.warn(
            "WARNING: Conditions are too loose to select countries that must be included in the data."
        )
        selected = pd.DataFrame({"country": [], "fraction": []})
    elif (len(selected) == 0) or (
        (len(selected) == len(reference)) and (len(reference) > 1)
    ):
        # This happens when the only way to fulfil the conditions is to include all countries.
        warnings.warn(
            "WARNING: Conditions are too strict to select countries that must be included in the data."
        )
        selected = reference.copy()

    print(
        f"{len(selected)} countries must be informed for {region} (covering {selected['fraction'].sum() * 100: .2f}% "
        f"of the population; otherwise aggregate data will be nan."
    )
    countries = selected["country"].tolist()

    return countries


def add_region_aggregates(
    df,
    region,
    countries_in_region=None,
    countries_that_must_have_data=None,
    num_allowed_nans_per_year=NUM_ALLOWED_NANS_PER_YEAR,
    frac_allowed_nans_per_year=FRAC_ALLOWED_NANS_PER_YEAR,
    country_col="Country",
    year_col="Year",
    aggregations=None,
):
    """Add data for regions (e.g. income groups or continents) to a dataset, or replace it, if the dataset already
    contains data for that region.

    When adding up the contribution from different countries (e.g. Spain, France, etc.) of a region (e.g. Europe), we
    want to avoid two problems:
    * Generating a series of nan, because one small country (with a negligible contribution) has nans.
    * Generating a series that underestimates the real one, because of treating missing values as zeros.

    To avoid these problems, we first define a list of "big countries" that must be present in the data, in order to
    safely do the aggregation. If any of these countries is not present for a particular variable and year, the
    aggregation will be nan for that variable and year. Otherwise, if all big countries are present, any other missing
    country will be assumed to have zero contribution to the variable.
    For example, when aggregating the electricity demand of North America, United States and Mexico cannot be missing,
    because otherwise the aggregation would significantly underestimate the true electricity demand of North America.

    Additionally, the aggregation of a particular variable for a particular year cannot have too many nans. If the
    number of nans exceeds num_allowed_nans_per_year, or if the fraction of nans exceeds frac_allowed_nans_per_year, the
    aggregation for that variable and year will be nan.

    Parameters
    ----------
    df : pd.Dataframe
        Original dataset, which may contain data for that region (in which case, it will be replaced by the ).
    region : str
        Region to add.
    countries_in_region : list or None
        List of countries that are members of this region. None to load them from countries-regions dataset.
    countries_that_must_have_data : list or None
        List of countries that must have data for a particular variable and year, otherwise the region will have nan for
        that particular variable and year. See function list_countries_in_region_that_must_have_data for more
        details.
    num_allowed_nans_per_year : int or None
        Maximum number of nans that can be present in a particular variable and year. If exceeded, the aggregation will
        be nan.
    frac_allowed_nans_per_year : float or None
        Maximum fraction of nans that can be present in a particular variable and year. If exceeded, the aggregation
        will be nan.
    country_col : str
        Name of country column.
    year_col : str
        Name of year column.
    aggregations : dict or None
        Aggregations to execute for each variable. If None, the contribution to each variable from each country in the
        region will be summed. Otherwise, only the variables indicated in the dictionary will be affected. All remaining
        variables will be nan.

    Returns
    -------
    df_updated : pd.DataFrame
        Original dataset after adding (or replacing) data for selected region.

    """
    if countries_in_region is None:
        # List countries in the region.
        countries_in_region = list_countries_in_region(
            region=region,
        )

    if countries_that_must_have_data is None:
        # List countries that should present in the data (since they are expected to contribute the most).
        countries_that_must_have_data = list_countries_in_region_that_must_have_data(
            region=region,
        )

    # If aggregations are not defined for each variable, assume 'sum'.
    fixed_columns = [country_col, year_col]
    if aggregations is None:
        aggregations = {
            variable: "sum" for variable in df.columns if variable not in fixed_columns
        }
    variables = list(aggregations)

    # Initialise dataframe of added regions, and add variables one by one to it.
    df_region = pd.DataFrame({country_col: [], year_col: []}).astype(
        dtype={country_col: "object", year_col: "int"}
    )
    # Select data for countries in the region.
    df_countries = df[df[country_col].isin(countries_in_region)]
    for variable in variables:
        df_clean = df_countries.dropna(subset=variable).reset_index(drop=True)
        df_added = groupby_agg(
            df=df_clean,
            groupby_columns=year_col,
            aggregations={
                country_col: lambda x: set(countries_that_must_have_data).issubset(
                    set(list(x))
                ),
                variable: aggregations[variable],
            },
            num_allowed_nans=num_allowed_nans_per_year,
            frac_allowed_nans=frac_allowed_nans_per_year,
        ).reset_index()
        # Make nan all aggregates if the most contributing countries were not present.
        df_added.loc[~df_added[country_col], variable] = np.nan
        # Replace the column that was used to check if most contributing countries were present by the region's name.
        df_added[country_col] = region
        # Include this variable to the dataframe of added regions.
        df_region = pd.merge(
            df_region, df_added, on=[country_col, year_col], how="outer"
        )

    # Remove rows in the original dataframe containing rows for region, and append new rows for region.
    df_updated = pd.concat(
        [df[~(df[country_col] == region)], df_region], ignore_index=True
    )

    # Sort conveniently.
    df_updated = df_updated.sort_values([country_col, year_col]).reset_index(drop=True)

    return df_updated


def multi_merge(dfs, on, how="inner"):
    """Merge multiple dataframes.

    This is a helper function when merging more than two dataframes on common columns.

    Parameters
    ----------
    dfs : list
        Dataframes to be merged.
    how : str
        Method to use for merging (with the same options available in pd.merge).
    on : list or str
        Column or list of columns on which to merge. These columns must have the same name on all dataframes.

    Returns
    -------
    merged : pd.DataFrame
        Input dataframes merged.

    """
    merged = dfs[0].copy()
    for df in dfs[1:]:
        merged = pd.merge(merged, df, how=how, on=on)

    return merged
