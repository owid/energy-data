"""Test functions in scripts.utils module.

"""

import json
from unittest.mock import patch, mock_open

import numpy as np
import pandas as pd

from scripts import utils


mock_countries = {
    "country_02": "Country_02",
    "country_03": "Country_03",
}

mock_population = pd.DataFrame(
    {
        "country": ["Country_01", "Country_01", "Country_02", "Country_02"],
        "year": [2020, 2021, 2019, 2020],
        "population": [10, 20, 30, 40],
    }
)


class MockPopulationLoad:
    def __init__(self, *args, **kwargs):
        self.population = mock_population

    def load(self):
        return self.population


class TestCompareDataFrames:
    def test_with_large_absolute_tolerance_all_equal(self):
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [1, 2]}),
            df2=pd.DataFrame({"col_01": [2, 3]}),
            absolute_tolerance=1,
            relative_tolerance=1e-8,
        ).equals(pd.DataFrame({"col_01": [True, True]}))

    def test_with_large_absolute_tolerance_all_unequal(self):
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [1, 2]}),
            df2=pd.DataFrame({"col_01": [2, 3]}),
            absolute_tolerance=0.9,
            relative_tolerance=1e-8,
        ).equals(pd.DataFrame({"col_01": [False, False]}))

    def test_with_large_absolute_tolerance_mixed(self):
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [1, 2]}),
            df2=pd.DataFrame({"col_01": [2, 3.1]}),
            absolute_tolerance=1,
            relative_tolerance=1e-8,
        ).equals(pd.DataFrame({"col_01": [True, False]}))

    def test_with_large_relative_tolerance_all_equal(self):
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [1, 2]}),
            df2=pd.DataFrame({"col_01": [2, 3]}),
            absolute_tolerance=1e-8,
            relative_tolerance=0.5,
        ).equals(pd.DataFrame({"col_01": [True, True]}))

    def test_with_large_relative_tolerance_all_unequal(self):
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [1, 2]}),
            df2=pd.DataFrame({"col_01": [2, 3]}),
            absolute_tolerance=1e-8,
            relative_tolerance=0.3,
        ).equals(pd.DataFrame({"col_01": [False, False]}))

    def test_with_large_relative_tolerance_mixed(self):
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [1, 2]}),
            df2=pd.DataFrame({"col_01": [2, 3]}),
            absolute_tolerance=1e-8,
            relative_tolerance=0.4,
        ).equals(pd.DataFrame({"col_01": [False, True]}))


class TestAreDataFramesEqual:
    def test_on_equal_dataframes_with_one_integer_column(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2, 3]}),
            df2=pd.DataFrame({"col_01": [1, 2, 3]}),
        )[0]

    def test_on_almost_equal_dataframes_but_differing_by_one_element(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2, 3]}),
            df2=pd.DataFrame({"col_01": [1, 2, 0]}),
        )[0]

    def test_on_almost_equal_dataframes_but_differing_by_type(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2, 3]}),
            df2=pd.DataFrame({"col_01": [1, 2, 3.0]}),
        )[0]

    def test_on_equal_dataframes_containing_nans(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2, np.nan]}),
            df2=pd.DataFrame({"col_01": [1, 2, np.nan]}),
        )[0]

    def test_on_equal_dataframes_containing_only_nans(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [np.nan, np.nan]}),
            df2=pd.DataFrame({"col_01": [np.nan, np.nan]}),
        )[0]

    def test_on_equal_dataframes_both_empty(self):
        assert utils.are_dataframes_equal(df1=pd.DataFrame(), df2=pd.DataFrame())[0]

    def test_on_equal_dataframes_with_various_types_of_columns(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame(
                {
                    "col_01": [1, 2],
                    "col_02": [0.1, 0.2],
                    "col_03": ["1", "2"],
                    "col_04": [True, False],
                }
            ),
            df2=pd.DataFrame(
                {
                    "col_01": [1, 2],
                    "col_02": [0.1, 0.2],
                    "col_03": ["1", "2"],
                    "col_04": [True, False],
                }
            ),
        )[0]

    def test_on_almost_equal_dataframes_but_columns_sorted_differently(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame(
                {
                    "col_01": [1, 2],
                    "col_02": [0.1, 0.2],
                    "col_03": ["1", "2"],
                    "col_04": [True, False],
                }
            ),
            df2=pd.DataFrame(
                {
                    "col_02": [0.1, 0.2],
                    "col_01": [1, 2],
                    "col_03": ["1", "2"],
                    "col_04": [True, False],
                }
            ),
        )[0]

    def test_on_unequal_dataframes_with_all_columns_different(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2], "col_02": [0.1, 0.2]}),
            df2=pd.DataFrame({"col_03": [0.1, 0.2], "col_04": [1, 2]}),
        )[0]

    def test_on_unequal_dataframes_with_some_common_columns(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2], "col_02": [0.1, 0.2]}),
            df2=pd.DataFrame({"col_01": [1, 2], "col_03": [1, 2]}),
        )[0]

    def test_on_equal_dataframes_given_large_absolute_tolerance(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [10, 20]}),
            df2=pd.DataFrame({"col_01": [11, 21]}),
            absolute_tolerance=1,
            relative_tolerance=1e-8,
        )[0]

    def test_on_unequal_dataframes_given_large_absolute_tolerance(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [10, 20]}),
            df2=pd.DataFrame({"col_01": [11, 21]}),
            absolute_tolerance=0.9,
            relative_tolerance=1e-8,
        )[0]

    def test_on_equal_dataframes_given_large_relative_tolerance(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1]}),
            df2=pd.DataFrame({"col_01": [2]}),
            absolute_tolerance=1e-8,
            relative_tolerance=0.5,
        )[0]

    def test_on_unequal_dataframes_given_large_relative_tolerance(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1]}),
            df2=pd.DataFrame({"col_01": [2]}),
            absolute_tolerance=1e-8,
            relative_tolerance=0.49,
        )[0]


@patch.object(utils.catalog, "find", MockPopulationLoad)
class TestAddPopulationToDataframe:
    def test_all_countries_and_years_in_population(self):
        df_in = pd.DataFrame(
            {"country": ["Country_02", "Country_01"], "year": [2019, 2021]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country_02", "Country_01"],
                "year": [2019, 2021],
                "population": [30, 20],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_countries_and_years_in_population_just_one(self):
        df_in = pd.DataFrame(
            {"country": ["Country_02", "Country_02"], "year": [2020, 2019]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country_02", "Country_02"],
                "year": [2020, 2019],
                "population": [40, 30],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_one_country_in_and_another_not_in_population(self):
        df_in = pd.DataFrame(
            {"country": ["Country_01", "Country_03"], "year": [2020, 2021]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country_01", "Country_03"],
                "year": [2020, 2021],
                "population": [10, np.nan],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_no_countries_in_population(self):
        df_in = pd.DataFrame(
            {"country": ["Country_04", "Country_04"], "year": [2000, 2000]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country_04", "Country_04"],
                "year": [2000, 2000],
                "population": [np.nan, np.nan],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_countries_in_population_but_not_for_given_years(self):
        df_in = pd.DataFrame(
            {"country": ["Country_02", "Country_01"], "year": [2000, 2000]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country_02", "Country_01"],
                "year": [2000, 2000],
                "population": [np.nan, np.nan],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_countries_in_population_but_a_year_in_and_another_not_in_population(self):
        df_in = pd.DataFrame(
            {"country": ["Country_02", "Country_01"], "year": [2019, 2000]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country_02", "Country_01"],
                "year": [2019, 2000],
                "population": [30, np.nan],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_change_country_and_year_column_names(self):
        df_in = pd.DataFrame(
            {"Country": ["Country_02", "Country_01"], "Year": [2019, 2021]}
        )
        df_out = pd.DataFrame(
            {
                "Country": ["Country_02", "Country_01"],
                "Year": [2019, 2021],
                "population": [30, 20],
            }
        )
        assert utils.add_population_to_dataframe(
            df=df_in, country_col="Country", year_col="Year"
        ).equals(df_out)


@patch("builtins.open", new_callable=mock_open, read_data=json.dumps(mock_countries))
class TestStandardizeCountries:
    def test_one_country_unchanged_and_another_changed(self, _):
        df_in = pd.DataFrame(
            {"country": ["Country_01", "country_02"], "some_variable": [1, 2]}
        )
        df_out = pd.DataFrame(
            {"country": ["Country_01", "Country_02"], "some_variable": [1, 2]}
        )
        assert utils.standardize_countries(df=df_in, countries_file="IGNORED").equals(
            df_out
        )

    def test_one_country_unchanged_and_another_unknown(self, _):
        df_in = pd.DataFrame(
            {"country": ["Country_01", "country_04"], "some_variable": [1, 2]}
        )
        df_out = pd.DataFrame(
            {"country": ["Country_01", "country_04"], "some_variable": [1, 2]}
        )
        assert utils.standardize_countries(df=df_in, countries_file="IGNORED").equals(
            df_out
        )

    def test_two_unknown_countries_made_nan(self, _):
        df_in = pd.DataFrame(
            {"country": ["Country_01", "country_04"], "some_variable": [1, 2]}
        )
        df_out = pd.DataFrame({"country": [np.nan, np.nan], "some_variable": [1, 2]})
        assert utils.are_dataframes_equal(
            df1=df_out,
            df2=utils.standardize_countries(
                df=df_in, countries_file="IGNORED", make_missing_countries_nan=True
            ),
        )

    def test_one_unknown_country_made_nan_and_a_known_country_changed(self, _):
        df_in = pd.DataFrame(
            {"country": ["Country_01", "country_02"], "some_variable": [1, 2]}
        )
        df_out = pd.DataFrame(
            {"country": [np.nan, "Country_02"], "some_variable": [1, 2]}
        )
        assert utils.are_dataframes_equal(
            df1=df_out,
            df2=utils.standardize_countries(
                df=df_in, countries_file="IGNORED", make_missing_countries_nan=True
            ),
        )

    def test_one_unknown_country_made_nan_a_known_country_changed_and_another_unchanged(
        self, _
    ):
        df_in = pd.DataFrame(
            {
                "country": ["Country_01", "country_02", "Country_03"],
                "some_variable": [1, 2, 3],
            }
        )
        df_out = pd.DataFrame(
            {
                "country": [np.nan, "Country_02", "Country_03"],
                "some_variable": [1, 2, 3],
            }
        )
        assert utils.are_dataframes_equal(
            df1=df_out,
            df2=utils.standardize_countries(
                df=df_in, countries_file="IGNORED", make_missing_countries_nan=True
            ),
        )

    def test_on_dataframe_with_no_countries(self, _):
        df_in = pd.DataFrame({"country": []})
        df_out = pd.DataFrame({"country": []})
        assert utils.are_dataframes_equal(
            df1=df_out,
            df2=utils.standardize_countries(df=df_in, countries_file="IGNORED"),
        )

    def test_change_country_column_name(self, _):
        df_in = pd.DataFrame({"Country": ["country_02"]})
        df_out = pd.DataFrame({"Country": ["Country_02"]})
        assert utils.are_dataframes_equal(
            df1=df_out,
            df2=utils.standardize_countries(
                df=df_in, countries_file="IGNORED", country_col="Country"
            ),
        )
