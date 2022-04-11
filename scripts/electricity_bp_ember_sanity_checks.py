"""Perform sanity checks on the Electricity mix from BP & Ember dataset, and compare the most recent version
of the dataset with the previous.

If the new dataset does not fulfil certain requirements, this script will raise an error.
If the new dataset does fulfil those requirements, but do not fully pass certain checks, it will create an HTML summary
of warnings and figures to be visually inspected.

Sanity checks performed on new dataset:
* Check that countries in the new dataset are in the population dataset (i.e. the input/shared/population.csv file).
* Check that all countries have data for a reasonable range of years.
* Check that all variables lie within an acceptable range of values.
* Check that all columns in the dataset have been inspected (by the previous checks).
Sanity checks that compare new with old dataset:
* Check that all columns coincide in both datasets.
* Check that all countries coincide in both datasets.
* Check that electricity share has not changed significantly on common data points from old to new dataset.
  * Points are compared only if at least one of them is large enough (to avoid large errors on insignificant values).
  * A figure is added to the output HTML file if the resulting error is larger than a certain threshold.
* Idem for total electricity.
* Idem for per capita electricity.

"""

import abc
import argparse
import base64
import os
import webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from owid import catalog
from tqdm.auto import tqdm

# Define common paths.

# Current folder.
CURRENT_DIR = os.path.dirname(__file__)
# Folder where datasets are stored.
GRAPHER_DATA_DIR = os.path.join(CURRENT_DIR, "grapher")
# Date tag and output file for visual inspection of potential issues with the dataset.
DATE_TAG = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = os.path.join(
    CURRENT_DIR, f"electricity_bp_ember_sanity_checks_{DATE_TAG}.html"
)
# Path to new dataset file.
CURRENT_YEAR = datetime.now().year
DATA_FILE_NEW = os.path.join(
    GRAPHER_DATA_DIR, f"Electricity mix from BP & EMBER ({CURRENT_YEAR}).csv"
)
# Path to old dataset file.
DATA_FILE_OLD = os.path.join(
    GRAPHER_DATA_DIR, f"Electricity mix from BP & EMBER ({CURRENT_YEAR - 1}).csv"
)

# Define parameters for output figures.

# Label to use for old dataset.
DATA_LABEL_OLD = "old"
# Label to use for new dataset.
DATA_LABEL_NEW = "new"
# True to include interactive plots in output HTML file (which can make the inspection slow if there are many figures);
# False to include plots as static images.
EXPORT_INTERACTIVE_PLOTS = False
# Maximum number of plots (of potentially problematic cases) to show in output file.
MAX_NUM_PLOTS = 150
# Range of acceptable values for electricity share (which is a percentage of total electricity).
MIN_ACCEPTED_VALUE_ELECTRICITY_SHARE = 0
MAX_ACCEPTED_VALUE_ELECTRICITY_SHARE = 100
# Range of acceptable values for total electricity.
# Here, the maximum will be the world maximum for each specific variable.
MIN_ACCEPTED_VALUE_ELECTRICITY_TOTAL = 0
# Range of acceptable values for per capita electricity.
MIN_ACCEPTED_VALUE_ELECTRICITY_PER_CAPITA = 0
MAX_ACCEPTED_VALUE_ELECTRICITY_PER_CAPITA = 60000
# MIN_RELEVANT_VALUE_* is the minimum value of a data point (from old or from new datasets) to be considered in the
# calculation of the error. This is used to avoid inspecting large errors on small quantities.
# E.g. 5 (in some units) means that errors will be calculated only when old or new variables are above 5.
MIN_RELEVANT_VALUE_ELECTRICITY_SHARE = 20
MIN_RELEVANT_VALUE_ELECTRICITY_TOTAL = 1
MIN_RELEVANT_VALUE_ELECTRICITY_PER_CAPITA = 100
# MIN_RELEVANT_ERROR is the minimum error (defined below) to consider as potentially problematic.
MIN_RELEVANT_ERROR = 30

# Define naming of entities in the different datasets used.

# Define default entity names.
NAME = {
    "country": "Country",
    "year": "Year",
    "electricity_generation": "Electricity generation (TWh)",
    "electricity_other_renewables": "Electricity from other renewables including bioenergy (TWh)",
    "electricity_share_renewables": "Other renewables including bioenergy (% electricity)",
    "electricity_other_renewable_per_capita": "Other renewable electricity including bioenergy per capita (kWh)",
}
# Define entity names for old dataset.
NAME_OLD = {
    "country": "Country",
    "year": "Year",
    "electricity_generation": "Electricity Generation (TWh)",
    "electricity_other_renewables": "Electricity from other renewables (TWh)",
    "electricity_share_renewables": "Other renewables (% electricity)",
    "electricity_other_renewable_per_capita": "Other renewable electricity per capita (kWh)",
}
# Define entity names for new dataset.
NAME_NEW = {
    "country": "Country",
    "year": "Year",
    "electricity_generation": "Electricity generation (TWh)",
    "electricity_other_renewables": "Electricity from other renewables including bioenergy (TWh)",
    "electricity_share_renewables": "Other renewables including bioenergy (% electricity)",
    "electricity_other_renewable_per_capita": "Other renewable electricity including bioenergy per capita (kWh)",
}
# Define entity names for OWID population dataset.
NAME_POPULATION = {
    "country": "country",
    "year": "year",
}
# Define columns (default entity names) related to electricity share.
COLUMNS_ELECTRICITY_SHARE = [
    "Coal (% electricity)",
    "Oil (% electricity)",
    "Gas (% electricity)",
    "Nuclear (% electricity)",
    "Solar (% electricity)",
    "Wind (% electricity)",
    "Hydro (% electricity)",
    "Other renewables including bioenergy (% electricity)",
    "Renewables (% electricity)",
    "Fossil fuels (% electricity)",
    "Low-carbon electricity (% electricity)",
    "Bioenergy (% electricity)",
    "Other renewables excluding bioenergy (% electricity)",
    "Electricity as share of primary energy",
]
# Define columns (default entity names) related to total electricity.
COLUMNS_ELECTRICITY_TOTAL = [
    "Electricity from coal (TWh)",
    "Electricity generation (TWh)",
    "Electricity from fossil fuels (TWh)",
    "Electricity from gas (TWh)",
    "Electricity from hydro (TWh)",
    "Low-carbon electricity (TWh)",
    "Electricity from nuclear (TWh)",
    "Electricity from oil (TWh)",
    "Electricity from other renewables including bioenergy (TWh)",
    "Electricity from renewables (TWh)",
    "Electricity from solar (TWh)",
    "Electricity from wind (TWh)",
    "Electricity from bioenergy (TWh)",
    "Electricity from other renewables excluding bioenergy (TWh)",
    "Electricity demand (TWh)",
]
# Define columns (default entity names) related to per capita electricity.
COLUMNS_ELECTRICITY_PER_CAPITA = [
    "Bioenergy electricity per capita (kWh)",
    "Coal electricity per capita (kWh)",
    "Fossil fuel electricity per capita (kWh)",
    "Gas electricity per capita (kWh)",
    "Hydro electricity per capita (kWh)",
    "Low-carbon electricity per capita (kWh)",
    "Nuclear electricity per capita (kWh)",
    "Oil electricity per capita (kWh)",
    "Other renewable electricity excluding bioenergy per capita (kWh)",
    "Other renewable electricity including bioenergy per capita (kWh)",
    "Per capita electricity (kWh)",
    "Renewable electricity per capita (kWh)",
    "Solar electricity per capita (kWh)",
    "Wind electricity per capita (kWh)",
]


def rename_columns(
    data, entities_to_rename_in_columns, name_dataset, name_default=None
):
    """Translate columns in a dataframe, from a naming convention to another.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    entities_to_rename_in_columns : list
        Entities (using default naming) that need to be translated in dataset.
    name_dataset : dict
        Dictionary of entity names in the dataset to be translated.
    name_default : dict
        Default entity names.

    Returns
    -------
    data_renamed : pd.DataFrame
        Dataset with columns adjusted to default naming.

    """
    if name_default is None:
        name_default = NAME
    columns_renaming = {}
    for entity in entities_to_rename_in_columns:
        # Ensure column exists in dictionary of entity names for considered dataset.
        error = f"ERROR: Entity {entity} is not defined in dictionary of entity names for this dataset."
        assert entity in name_dataset, error

        # Ensure column exists in default dictionary of entity names.
        error = f"ERROR: Entity {entity} is not defined in default dictionary of entity names."
        assert entity in name_default, error

        columns_renaming[name_dataset[entity]] = name_default[entity]

    data_renamed = data.rename(columns=columns_renaming)

    return data_renamed


def load_old_data(data_file_old=DATA_FILE_OLD, name_old=None):
    """Load old dataset and return it with the default entity naming.

    Parameters
    ----------
    data_file_old : str
        Path to old dataset file.
    name_old : dict
        Dictionary of entity names in the old dataset.

    Returns
    -------
    data_old : pd.DataFrame
        Old dataset.

    """
    if name_old is None:
        name_old = NAME_OLD
    entities_to_rename_in_columns = [
        "country",
        "year",
        "electricity_generation",
        "electricity_other_renewables",
        "electricity_share_renewables",
        "electricity_other_renewable_per_capita",
    ]
    data_old_raw = pd.read_csv(data_file_old)
    data_old = rename_columns(
        data=data_old_raw,
        entities_to_rename_in_columns=entities_to_rename_in_columns,
        name_dataset=name_old,
    )

    return data_old


def load_new_data(data_file_new=DATA_FILE_NEW, name_new=None):
    """Load new dataset and return it with the default entity naming.

    Parameters
    ----------
    data_file_new : str
        Path to new dataset file.
    name_new : dict
        Dictionary of entity names in the new dataset.

    Returns
    -------
    data_new : pd.DataFrame
        New dataset.

    """
    if name_new is None:
        name_new = NAME_NEW
    entities_to_rename_in_columns = [
        "country",
        "year",
        "electricity_generation",
        "electricity_other_renewables",
        "electricity_share_renewables",
        "electricity_other_renewable_per_capita",
    ]
    data_new_raw = pd.read_csv(data_file_new)
    data_new = rename_columns(
        data=data_new_raw,
        entities_to_rename_in_columns=entities_to_rename_in_columns,
        name_dataset=name_new,
    )

    return data_new


def load_population(name_population=None):
    """Load population dataset and return it with the default entity naming.

    Parameters
    ----------
    name_population : dict
        Dictionary of entity names in the population dataset.

    Returns
    -------
    population_renamed : pd.DataFrame
        Population dataset.

    """
    if name_population is None:
        name_population = NAME_POPULATION
    population = (
        catalog.find("population", namespace="owid", dataset="key_indicators")
        .load()
        .reset_index()
    )
    population_renamed = rename_columns(
        data=population,
        entities_to_rename_in_columns=["country", "year"],
        name_dataset=name_population,
    )

    return population_renamed


def mean_absolute_percentage_error(old, new, epsilon=1e-6):
    """Mean absolute percentage error (MAPE).

    Parameters
    ----------
    old : pd.Series
        Old values.
    new : pd.Series
        New values.
    epsilon : float
        Small number that avoids divisions by zero.

    Returns
    -------
    error : float
        MAPE.

    """
    error = np.mean(abs(new - old) / (old + epsilon)) * 100

    return error


def detect_variables_outside_expected_ranges(
    data, columns, min_value, max_value, name=None
):
    """Detect variables in dataset that have values outside an expected range.

    Raise a warning for every country-variable case where this happens.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    columns : list
        Columns (variables of the dataset) to check.
    min_value : float or int
        Minimum expected value.
    max_value : float or int
        Maximum expected value.
    name : dict
        Default entity names.

    Returns
    -------
    summary : str
        HTML with a summary of the results of this check.

    """
    if name is None:
        name = NAME
    incorrect = data.melt(
        id_vars=[name["country"], name["year"]], var_name="Variable", value_name="Value"
    )
    incorrect = incorrect[
        (incorrect["Variable"].isin(columns))
        & ((incorrect["Value"] < min_value) | (incorrect["Value"] > max_value))
    ].reset_index(drop=True)
    incorrect = (
        incorrect.groupby([name["country"], "Variable"])
        .agg({"Value": max})
        .sort_values("Value", ascending=False)
        .reset_index()
    )

    return incorrect


class Check(abc.ABC):
    """Common abstract check."""

    def __init__(self):
        self.num_warnings = 0

    def apply_all_checks(self):
        """Apply all methods in the class that are called 'check_', assume they do not ingest any argument, and execute
        them one by one, gathering all their output summaries.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        all_checks = [
            check
            for check in dir(self)
            if callable(getattr(self, check))
            if check.startswith("check_")
        ]
        summary = ""
        # Ensure the count of warnings starts from zero.
        self.num_warnings = 0
        for i, check in enumerate(tqdm(all_checks)):
            print(f"* Check ({i + 1}/{len(all_checks)}): {check}")
            summary += getattr(self, check)()
        if self.num_warnings > 0:
            print(
                f"There were {self.num_warnings} warnings generated. Visually inspect sanity checks file."
            )
        else:
            print("All checks passed without any warning.")

        return summary


class SanityChecksOnSingleDataset(Check):
    def __init__(
        self,
        data,
        name=None,
        columns_electricity_share=None,
        columns_electricity_total=None,
        columns_electricity_per_capita=None,
        min_accepted_value_electricity_share=MIN_ACCEPTED_VALUE_ELECTRICITY_SHARE,
        max_accepted_value_electricity_share=MAX_ACCEPTED_VALUE_ELECTRICITY_SHARE,
        min_accepted_value_electricity_total=MIN_ACCEPTED_VALUE_ELECTRICITY_TOTAL,
        min_accepted_value_electricity_per_capita=MIN_ACCEPTED_VALUE_ELECTRICITY_PER_CAPITA,
        max_accepted_value_electricity_per_capita=MAX_ACCEPTED_VALUE_ELECTRICITY_PER_CAPITA,
    ):
        """Sanity checks to apply to a single dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset.
        name : dict
            Default entity names.
        columns_electricity_share : list
            Columns (variables) in dataset corresponding to electricity share variables.
        columns_electricity_total : list
            Columns (variables) in dataset corresponding to total electricity variables.

        """
        super().__init__()
        if name is None:
            name = NAME
        if columns_electricity_share is None:
            columns_electricity_share = COLUMNS_ELECTRICITY_SHARE
        if columns_electricity_total is None:
            columns_electricity_total = COLUMNS_ELECTRICITY_TOTAL
        if columns_electricity_per_capita is None:
            columns_electricity_per_capita = COLUMNS_ELECTRICITY_PER_CAPITA
        self.data = data
        self.name = name
        self.columns_electricity_share = columns_electricity_share
        self.columns_electricity_total = columns_electricity_total
        self.columns_electricity_per_capita = columns_electricity_per_capita
        self.min_accepted_value_electricity_share = min_accepted_value_electricity_share
        self.max_accepted_value_electricity_share = max_accepted_value_electricity_share
        self.min_accepted_value_electricity_total = min_accepted_value_electricity_total
        self.min_accepted_value_electricity_per_capita = (
            min_accepted_value_electricity_per_capita
        )
        self.max_accepted_value_electricity_per_capita = (
            max_accepted_value_electricity_per_capita
        )
        # Load population dataset.
        self.population = load_population()

    def check_that_countries_are_in_population_dataset(self):
        """Check that countries/regions in dataset are included in the population dataset.

        Raise Warnings for each country/region that is not in population dataset.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        summary = "<br><br> Check that all countries in new dataset are in the OWID population dataset."
        missing_in_population = set(self.data[self.name["country"]]) - set(
            self.population[self.name["country"]]
        )
        if len(missing_in_population) > 0:
            summary += (
                f"<br><font color='red'>WARNING: {len(missing_in_population)} countries/regions not found in "
                f"population dataset."
            )
            summary += "".join(
                [f"<li> {country}.</li>" for country in missing_in_population]
            )
            summary += "</font>"
            self.num_warnings += 1

        return summary

    def check_year_ranges(
        self, min_year_latest_possible=2000, max_year_maximum_delay=3
    ):
        """Check that dataset has variables within a reasonable range of years.

        Parameters
        ----------
        min_year_latest_possible : int
            If the earliest data point occurs at a year that is later than this, a Warning is raised.
        max_year_maximum_delay : int
            If the latest data point occurs earlier than this number of years ago, a Warning is raised.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        summary = "<br><br> Check that all countries have at least one data point for each year in a reasonable range."
        # Keep only rows for which we have at least one not null data point.
        data_clean = self.data.dropna(how="all")
        year_ranges = (
            data_clean.groupby(self.name["country"])
            .agg({self.name["year"]: (min, max)})[self.name["year"]]
            .reset_index()
        )

        current_year = datetime.today().year
        # Check if minimum year is acceptable.
        minimum_year_too_recent = year_ranges[
            year_ranges["min"] > min_year_latest_possible
        ]
        if len(minimum_year_too_recent) > 0:
            summary += (
                f"<br><font color='red'>WARNING: Minimum year is more recent than {min_year_latest_possible} "
                + f"for {len(minimum_year_too_recent)} countries/regions."
            )
            summary += "</font>"
            self.num_warnings += 1

        # Check if maximum year is acceptable.
        selected_countries = []
        year_limit = current_year
        for delay in range(max_year_maximum_delay + 1):
            year_limit = current_year - delay
            selected_countries = year_ranges[year_ranges["max"] < year_limit]
        if len(selected_countries) > 0:
            summary += (
                f"<br><font color='red'>WARNING: {len(selected_countries)} countries/regions with latest "
                f"data prior to {year_limit}:"
            )
            summary += "".join(
                [
                    f"<li> {country}.</li>"
                    for country in selected_countries[self.name["country"]].tolist()
                ]
            )
            summary += "</font>"
            self.num_warnings += 1

        return summary

    def _generate_warning_for_variables_outside_expected_ranges(self, incorrect):
        summary = ""
        if len(incorrect) > 0:
            summary += (
                f"<br><font color='red'>WARNING: {len(set(incorrect[self.name['country']]))} countries with "
                f"values outside expected range:"
            )
            for i, row in incorrect.iterrows():
                summary += f"<li> {row[self.name['country']]} - {row['Variable']}. Value: {row['Value']:.2f}"
            summary += "</font>"
            self.num_warnings += 1

        return summary

    def _check_electricity_share_ranges(self):
        summary = (
            f"<br><br> Check that share electricity is within expected range "
            f"({self.min_accepted_value_electricity_share} to {self.max_accepted_value_electricity_share})."
        )
        incorrect = detect_variables_outside_expected_ranges(
            data=self.data,
            columns=self.columns_electricity_share,
            min_value=self.min_accepted_value_electricity_share,
            max_value=self.max_accepted_value_electricity_share,
        )
        summary += self._generate_warning_for_variables_outside_expected_ranges(
            incorrect
        )

        return summary

    def _check_electricity_total_ranges(self):
        summary = ""
        for column in self.columns_electricity_total:
            world_maximum = self.data[self.data[self.name["country"]] == "World"][
                column
            ].max()
            summary += (
                f"<br> Check {self.min_accepted_value_electricity_total} < electricity < {int(world_maximum)} "
                f"(world's maximum) for: {column}"
            )
            incorrect = detect_variables_outside_expected_ranges(
                data=self.data,
                columns=[column],
                min_value=self.min_accepted_value_electricity_total,
                max_value=world_maximum,
            )
            summary += self._generate_warning_for_variables_outside_expected_ranges(
                incorrect
            )

        return summary

    def _check_electricity_per_capita_ranges(self):
        summary = (
            f"<br><br> Check that no country has electricity per capita below "
            f"{self.min_accepted_value_electricity_per_capita} or above "
            f"{self.max_accepted_value_electricity_per_capita}."
        )
        incorrect = detect_variables_outside_expected_ranges(
            data=self.data,
            columns=self.columns_electricity_per_capita,
            min_value=self.min_accepted_value_electricity_per_capita,
            max_value=self.max_accepted_value_electricity_per_capita,
        )
        summary += self._generate_warning_for_variables_outside_expected_ranges(
            incorrect
        )

        return summary

    def check_that_all_variables_are_in_acceptable_ranges(self):
        """Check that all variables (including electricity share, total and per capita) in dataset lie within acceptable
        ranges.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        summary_electricity_share = self._check_electricity_share_ranges()
        summary_electricity_total = self._check_electricity_total_ranges()
        summary_electricity_per_capita = self._check_electricity_per_capita_ranges()
        summary = (
            summary_electricity_share
            + summary_electricity_total
            + summary_electricity_per_capita
        )

        return summary

    def check_all_relevant_columns_inspected(self):
        """Check that all relevant columns in dataset have been inspected in other checks.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        summary = "<br><br> Check that all relevant columns have been inspected."
        unchecked_columns = list(
            set(self.data.columns)
            - set(
                self.columns_electricity_share
                + self.columns_electricity_total
                + self.columns_electricity_per_capita
                + [self.name["country"], self.name["year"]]
            )
        )
        if len(unchecked_columns) > 0:
            summary += f"<br><font color='red'>WARNING: Some columns were not included in any sanity check:"
            summary += "".join([f"<li> {column}.</li>" for column in unchecked_columns])
            summary += "</font>"
            self.num_warnings += 1

        return summary


class SanityChecksComparingTwoDatasets(Check):
    def __init__(
        self,
        data_old,
        data_new,
        name=None,
        data_label_old=DATA_LABEL_OLD,
        data_label_new=DATA_LABEL_NEW,
        columns_electricity_share=None,
        columns_electricity_total=None,
        columns_electricity_per_capita=None,
        min_relevant_value_electricity_share=MIN_RELEVANT_VALUE_ELECTRICITY_SHARE,
        min_relevant_value_electricity_total=MIN_RELEVANT_VALUE_ELECTRICITY_TOTAL,
        min_relevant_value_electricity_per_capita=MIN_RELEVANT_VALUE_ELECTRICITY_PER_CAPITA,
        min_relevant_error=MIN_RELEVANT_ERROR,
    ):
        """Sanity checks comparing a new dataset with an old one.

        Parameters
        ----------
        data_old : pd.DataFrame
            Old dataset.
        data_new : pd.DataFrame
            New dataset.
        name : dict
            Default entity names.
        columns_electricity_share : list
            Columns (variables) in dataset corresponding to electricity share variables.
        columns_electricity_total : list
            Columns (variables) in dataset corresponding to total electricity variables.

        """
        super().__init__()
        if name is None:
            name = NAME
        if columns_electricity_share is None:
            columns_electricity_share = COLUMNS_ELECTRICITY_SHARE
        if columns_electricity_total is None:
            columns_electricity_total = COLUMNS_ELECTRICITY_TOTAL
        if columns_electricity_per_capita is None:
            columns_electricity_per_capita = COLUMNS_ELECTRICITY_PER_CAPITA
        self.data_old = data_old
        self.data_new = data_new
        self.data_label_old = data_label_old
        self.data_label_new = data_label_new
        self.name = name
        self.columns_electricity_share = columns_electricity_share
        self.columns_electricity_total = columns_electricity_total
        self.columns_electricity_per_capita = columns_electricity_per_capita
        self.min_relevant_value_electricity_share = min_relevant_value_electricity_share
        self.min_relevant_value_electricity_total = min_relevant_value_electricity_total
        self.min_relevant_value_electricity_per_capita = (
            min_relevant_value_electricity_per_capita
        )
        self.min_relevant_error = min_relevant_error
        # Load population dataset.
        self.population = load_population()
        # Create comparison dataframe.
        self.comparison = self._create_comparison_dataframe()
        # Initialise dataframe of potential problems.
        self.problems = None
        # Initialise warnings count.
        self.num_warnings = 0

    def _create_comparison_dataframe(self):
        data_old_prepared = self.data_old.copy()
        data_old_prepared["source"] = self.data_label_old
        data_new_prepared = self.data_new.copy()
        data_new_prepared["source"] = self.data_label_new
        comparison = pd.concat(
            [data_old_prepared, data_new_prepared], ignore_index=True
        )

        return comparison

    def check_that_all_countries_are_in_both_datasets(self):
        """Check that all countries in the old dataset are in the new, and vice versa.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        summary = (
            "<br><br> Check that all countries in old dataset are in the new dataset."
        )
        missing_in_new = set(self.data_old[self.name["country"]]) - set(
            self.data_new[self.name["country"]]
        )
        if len(missing_in_new) > 0:
            summary += (
                f"<br><font color='red'>WARNING: {len(missing_in_new)} countries/regions in old dataset were "
                f"not found in new dataset."
            )
            summary += "".join([f"<li> {country}.</li>" for country in missing_in_new])
            summary += "<br></font>"
            self.num_warnings += 1
        summary += (
            "<br><br> Check that all countries in new dataset are in the old dataset."
        )
        missing_in_old = set(self.data_new[self.name["country"]]) - set(
            self.data_old[self.name["country"]]
        )
        if len(missing_in_old) > 0:
            summary += (
                f"<br><font color='red'>WARNING: {len(missing_in_old)} countries/regions in new dataset were "
                f"not found in old dataset."
            )
            summary += "".join([f"<li> {country}.</li>" for country in missing_in_old])
            summary += "<br></font>"
            self.num_warnings += 1

        return summary

    def check_columns_in_datasets_coincide(self):
        """Check that columns in old and new datasets coincide.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        # Compare columns in old and new dataset.
        variables_old = [
            column
            for column in self.data_old.columns
            if column not in [self.name["country"], self.name["year"]]
        ]
        variables_new = [
            column
            for column in self.data_new.columns
            if column not in [self.name["country"], self.name["year"]]
        ]
        variables_missing_in_data_new = list(set(variables_old) - set(variables_new))
        variables_added_in_data_new = list(set(variables_new) - set(variables_old))

        summary = f"<br><br> Check that all columns in old dataset are in new dataset."
        if len(variables_missing_in_data_new) > 0:
            summary += f"<br><font color='red'>WARNING: There are variables in old dataset that are not in new:"
            summary += "".join(
                [f"<li> {variable}.</li>" for variable in variables_missing_in_data_new]
            )
            summary += "<br></font>"
            self.num_warnings += 1
        summary += f"<br><br> Check that all columns in new dataset are in old dataset."
        if len(variables_added_in_data_new) > 0:
            summary += f"<br><font color='red'>WARNING: There are variables in new dataset that are not in old:"
            summary += "".join(
                [f"<li> {variable}.</li>" for variable in variables_added_in_data_new]
            )
            summary += "</font>"
            self.num_warnings += 1

        return summary

    def plot_time_series_for_country_and_variable(self, country, variable):
        """Plot a time series for a specific country and variable in the old dataset and an analogous time series for
        the new dataset.

        Parameters
        ----------
        country : str
            Country.
        variable : str
            Entity name for the variable to plot.

        Returns
        -------
        fig : plotly.Figure
            Plot.

        """
        # Select data for country.
        comparison = self.comparison[
            self.comparison[self.name["country"]] == country
        ].reset_index(drop=True)[[self.name["year"], variable, "source"]]
        # Add columns for plotting parameters.
        comparison["size"] = 0.003
        comparison.loc[comparison["source"] == self.data_label_new, "size"] = 0.001
        # hover_data = {'source': False, name['year']: False, variable: True, 'size': False}
        hover_data = {}

        fig = (
            px.scatter(
                comparison,
                x=self.name["year"],
                y=variable,
                color="source",
                size="size",
                size_max=10,
                color_discrete_sequence=["red", "green"],
                opacity=0.9,
                hover_name=self.name["year"],
                hover_data=hover_data,
            )
            .update_xaxes(
                showgrid=True,
                title="Year",
                autorange=False,
                range=[
                    comparison[self.name["year"]].min() - 1,
                    comparison[self.name["year"]].max() + 1,
                ],
            )
            .update_yaxes(
                showgrid=True,
                title=variable,
                autorange=False,
            )
            .update_layout(
                clickmode="event+select", autosize=True, title=f"{country} - {variable}"
            )
            .update_layout(font={"size": 9})
        )

        if "%" in variable:
            fig = fig.update_yaxes(range=[0, 100])
        else:
            fig = fig.update_yaxes(
                range=[
                    comparison[variable].min() * 0.9,
                    comparison[variable].max() * 1.1,
                ],
            )

        return fig

    def get_error_for_all_countries_and_specific_variables(
        self,
        variables,
        min_relevant_value=0,
        error_name="mape",
        error_metric=mean_absolute_percentage_error,
    ):
        """Compute the deviation between old and new datasets (by means of an error metric) for all countries and a list
        of specific variables.

        We expect the new dataset to not deviate much from the old one. Therefore, we compute the deviation between
        both, to detect potential issues with any of the datasets.

        Parameters
        ----------
        variables : list
            Variables (given by their default entity names) to be compared.
        min_relevant_value : float or int
            Minimum value for a variable to be considered relevant. Calculate errors only on those points where either
            the old or the new dataset passes this threshold.
        error_name : str
            Name for the error metric.
        error_metric : function
            Error metric. This function must ingest an 'old' and 'new' arguments.

        Returns
        -------
        errors : pd.DataFrame
            Table of countries and variables, and their respective deviations (in terms of the given error metric).

        """
        errors = pd.DataFrame()
        for country in tqdm(self.comparison[self.name["country"]].unique().tolist()):
            for variable in variables:
                comparison_pivot = (
                    self.comparison[self.comparison[self.name["country"]] == country]
                    .pivot(index=self.name["year"], columns="source", values=variable)
                    .dropna(how="any")
                    .reset_index()
                )
                for source in [self.data_label_old, self.data_label_new]:
                    if source not in comparison_pivot:
                        comparison_pivot[source] = np.nan
                # Omit rows where both old and new values are too small (to avoid large errors on irrelevant values).
                comparison_pivot = comparison_pivot[
                    (comparison_pivot[self.data_label_old] > min_relevant_value)
                    | (comparison_pivot[self.data_label_new] > min_relevant_value)
                ].reset_index(drop=True)
                error = error_metric(
                    old=comparison_pivot[self.data_label_old],
                    new=comparison_pivot[self.data_label_new],
                )
                errors = pd.concat(
                    [
                        errors,
                        pd.DataFrame(
                            {
                                self.name["country"]: [country],
                                "Variable": [variable],
                                error_name: [error],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

        # Compare errors with mean value of a variables.
        mean_variable_value = (
            self.comparison.groupby([self.name["country"]])
            .mean()
            .reset_index()
            .drop(columns=self.name["year"])
            .melt(
                id_vars=self.name["country"],
                var_name="Variable",
                value_name="mean_variable_value",
            )
        )
        mean_variable_value = mean_variable_value[
            mean_variable_value["Variable"].isin(variables)
        ].reset_index(drop=True)
        errors = pd.merge(
            mean_variable_value,
            errors,
            on=[self.name["country"], "Variable"],
            how="inner",
        )

        return errors

    def plot_all_metrics_for_country(self, country, variables):
        """Plot all metrics for a given country.

        Parameters
        ----------
        country : str
            Country to consider.
        variables : list
            Variables (given by their default entity names) to be compared.

        """
        for variable in variables:
            fig = self.plot_time_series_for_country_and_variable(
                country=country, variable=variable
            )
            fig.show()

    def save_comparison_plots_for_specific_variables_of_country(
        self, country, variables, output_file
    ):
        """Save plots comparing the old and new time series of a list of variables, for a specific country, in an HTML
        file.

        For example, to save all comparisons of a list of variables for a specific country:
        save_comparison_plots_for_specific_variables_of_country(
            comparison=comparison, country='United Kingdom',
            variables=['Electricity from solar (TWh)', 'Electricity from wind (TWh)'])

        Parameters
        ----------
        country : str
            Country to consider.
        variables : list
            Variables (given by their default entity names) to be compared.
        output_file : str
            Path to output HTML file.

        """
        # Ensure output folder exists.
        output_dir = os.path.dirname(output_file)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        # Create an HTML file with all figures.
        with open(output_file, "w") as f:
            for variable in variables:
                fig = self.plot_time_series_for_country_and_variable(
                    country=country, variable=variable
                )
                f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    def plot_comparison_for_problematic_cases(
        self,
        max_num_plots=MAX_NUM_PLOTS,
        export_interactive_plots=EXPORT_INTERACTIVE_PLOTS,
    ):
        """Plot time series from the old and new dataset, in certain, potentially problematic cases (country-variables),
        and return all these plots together as a string (HTML or base64).

        Parameters
        ----------
        max_num_plots : int
            Maximum number of plots to include. Raise a warning if the number of problematic cases is larger than this.
        export_interactive_plots : bool
            True to export figures as interactive plotly figures (which, if there are many, can slow down the inspection
            of figures). False to export figures encoded in base64 format.

        Returns
        -------
        figures : str
            Figures, placed one after the other, in HTML or base64 format.

        """
        if self.problems is None:
            self.problems = self.gather_potential_problems()
        figures = ""
        if len(self.problems) > max_num_plots:
            figures += (
                f"<br><font color='red'>WARNING: {len(self.problems)} figures to plot, only {max_num_plots} "
                f"will be shown.</font><br>"
            )
            self.num_warnings += 1
        for i, problem in tqdm(self.problems.iterrows(), total=len(self.problems)):
            fig = self.plot_time_series_for_country_and_variable(
                country=problem[self.name["country"]], variable=problem["Variable"]
            )
            if "mape" in problem:
                fig.update_layout(
                    title=f"{fig.layout.title['text']} - Relevant MAPE: {problem['mape']} % "
                )
            if export_interactive_plots:
                figures += fig.to_html(full_html=False, include_plotlyjs="cdn")
            else:
                img = plotly.io.to_image(fig, scale=1.2)
                img_base64 = base64.b64encode(img).decode("utf8")
                figures += (
                    f"<br><img class='icon' src='data:image/png;base64,{img_base64}'>"
                )

        return figures

    def detect_abrupt_changes_in_data(
        self, columns, min_relevant_value, min_error, error_name="mape"
    ):
        """Detect abrupt changes in variables when going from the old to the new dataset.

        Return a dataframe with all country-variables that are potentially problematic, since they changed too much in
        the new dataset with respect to the previous.

        Parameters
        ----------
        columns : list
        min_relevant_value : float or int
            Minimum value for a variable to be considered relevant. Calculate errors only on those points where either
            the old or the new dataset passes this threshold.
        min_error : float or int
            Minimum relevant error. Only country-variables that surpass this error will be considered.
        error_name : str
            Name of error metric.

        Returns
        -------
        problems : pd.DataFrame
            Potentially problematic cases of country-variable that has suffered a significant change.

        """
        # Calculate errors between old and new time series.
        errors = self.get_error_for_all_countries_and_specific_variables(
            variables=columns, min_relevant_value=min_relevant_value
        )
        errors[error_name] = errors[error_name].round(1)

        # Visually inspect any variable with a large error.
        problems = errors[(errors[error_name] > min_error)].sort_values(
            error_name, ascending=False
        )

        return problems

    def _generate_summary_for_checks_on_abrupt_changes(
        self, problems, min_relevant_value, min_error, electricity_type
    ):
        summary = f"<br><br> Check that {electricity_type} did not change abruptly between old and new dataset, where:"
        summary += f"<li> At least one of the two values compared is larger than {min_relevant_value}.</li>"
        summary += f"<li> The resulting error is larger than {min_error}.</li>"
        if len(problems) > 0:
            summary += (
                f"<br><font color='red'>WARNING: Data for {len(problems['Country'].unique())} countries "
                f"changed significantly:"
            )
            self.num_warnings += 1
            for i, row in problems.iterrows():
                summary += f"<li> {row['Country']:<30} - {row['Variable']:<55}. MAPE: {round(row['mape']):>2}%.</li>"
            summary += "</font>"

        return summary

    def gather_potential_problems(self):
        """Gather all country-variables that may potentially be problematic, because having changed too abruptly.

        Returns
        -------
        problems : pd.DataFrame
            Potentially problematic cases of country-variable that has suffered a significant change.

        """
        print(f"Gathering potential problems related to electricity share.")
        problems_electricity_share = self.detect_abrupt_changes_in_data(
            columns=self.columns_electricity_share,
            min_relevant_value=self.min_relevant_value_electricity_share,
            min_error=self.min_relevant_error,
        )
        print(f"Gathering potential problems related to total electricity.")
        problems_electricity_total = self.detect_abrupt_changes_in_data(
            columns=self.columns_electricity_total,
            min_relevant_value=self.min_relevant_value_electricity_total,
            min_error=self.min_relevant_error,
        )
        print(f"Gathering potential problems related to per capita electricity.")
        problems_electricity_per_capita = self.detect_abrupt_changes_in_data(
            columns=self.columns_electricity_per_capita,
            min_relevant_value=self.min_relevant_value_electricity_per_capita,
            min_error=self.min_relevant_error,
        )
        problems = pd.concat(
            [
                problems_electricity_share,
                problems_electricity_total,
                problems_electricity_per_capita,
            ],
            ignore_index=True,
        )

        return problems

    def check_that_variables_do_not_change_abruptly(self):
        """Check that variable values do not change significantly between the old and new datasets.

        Returns
        -------
        summary : str
            HTML with a summary of the results of this check.

        """
        if self.problems is None:
            self.problems = self.gather_potential_problems()
        problems_electricity_share = self.problems[
            self.problems["Variable"].isin(self.columns_electricity_share)
        ]
        problems_electricity_total = self.problems[
            self.problems["Variable"].isin(self.columns_electricity_total)
        ]
        problems_electricity_per_capita = self.problems[
            self.problems["Variable"].isin(self.columns_electricity_per_capita)
        ]
        summary_electricity_share = self._generate_summary_for_checks_on_abrupt_changes(
            problems=problems_electricity_share,
            min_relevant_value=self.min_relevant_value_electricity_share,
            min_error=self.min_relevant_error,
            electricity_type="electricity share",
        )
        summary_electricity_total = self._generate_summary_for_checks_on_abrupt_changes(
            problems=problems_electricity_total,
            min_relevant_value=self.min_relevant_value_electricity_total,
            min_error=self.min_relevant_error,
            electricity_type="total electricity",
        )
        summary_electricity_per_capita = (
            self._generate_summary_for_checks_on_abrupt_changes(
                problems=problems_electricity_per_capita,
                min_relevant_value=self.min_relevant_value_electricity_per_capita,
                min_error=self.min_relevant_error,
                electricity_type="per capita electricity",
            )
        )
        summary = (
            summary_electricity_share
            + summary_electricity_total
            + summary_electricity_per_capita
        )

        return summary


def main(
    data_file_old=DATA_FILE_OLD, data_file_new=DATA_FILE_NEW, output_file=OUTPUT_FILE
):
    """Apply all sanity checks and store the result as an HTML file to be visually inspected.

    Parameters
    ----------
    data_file_old : str
        Path to old dataset file.
    data_file_new : str
        Path to new dataset file.
    output_file : str
        Path to output HTML file to be visually inspected.

    """
    print("Loading data")
    data_old = load_old_data(data_file_old=data_file_old)
    data_new = load_new_data(data_file_new=data_file_new)

    print("Performing sanity checks on new dataset.")
    single_dataset_checks = SanityChecksOnSingleDataset(data=data_new)
    summary = single_dataset_checks.apply_all_checks()

    print("Performing sanity checks comparing to previous dataset.")
    comparison_checks = SanityChecksComparingTwoDatasets(
        data_old=data_old, data_new=data_new
    )
    summary += comparison_checks.apply_all_checks()

    print("Preparing final summary of figures to be visually inspected.")
    figures = comparison_checks.plot_comparison_for_problematic_cases()
    summary += figures

    print(f"Saving summary to file {output_file}.")
    with open(output_file, "w") as output_file_:
        output_file_.write(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform sanity checks on the Electricity mix from BP & Ember dataset, and compare the most recent "
        "version of the dataset with the previous."
    )
    parser.add_argument(
        "-f",
        "--output_file",
        default=OUTPUT_FILE,
        help=f"Path to output HTML file to be visually inspected. Default: "
        f"{OUTPUT_FILE}",
    )
    parser.add_argument(
        "-new",
        "--data_file_new",
        default=DATA_FILE_NEW,
        help=f"Path to new dataset file. Default: {DATA_FILE_NEW}",
    )
    parser.add_argument(
        "-old",
        "--data_file_old",
        default=DATA_FILE_OLD,
        help=f"Path to old dataset file. Default: {DATA_FILE_OLD}",
    )
    parser.add_argument(
        "-s",
        "--show_in_browser",
        default=False,
        action="store_true",
        help="If given, display output file in browser.",
    )
    args = parser.parse_args()

    main(
        data_file_old=args.data_file_old,
        data_file_new=args.data_file_new,
        output_file=args.output_file,
    )
    if args.show_in_browser:
        webbrowser.open("file://" + args.output_file)


# Conclusions and latest manual corrections (February 2022):
# * Electricity as share of primary energy was fixed (the unit of the source changed from Mtoe to EJ).
# * Moldova has been removed from the dataset, since there are clear inconsistencies between global and European data.
# * Some regions have been removed (e.g. 'North America'), since their definition by the original source did not match
#   our current OWID definition.
# * By visually inspecting all other countries we conclude that all deviations are due to inconsistencies between
#   BP and Ember datasets, or between global and European Ember datasets.
#   The new dataset tends to improve on the previous (it presents less abrupt changes over time).
