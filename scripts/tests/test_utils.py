"""Test functions in scripts.utils module.

"""

import json
import unittest
from unittest.mock import patch, mock_open

import numpy as np
import pandas as pd

from scripts import utils


mock_countries = {
    "country_02": "Country 2",
    "country_03": "Country 3",
}

mock_population = pd.DataFrame(
    {
        "country": ["Country 1", "Country 1", "Country 2", "Country 2", "Country 3"],
        "year": [2020, 2021, 2019, 2020, 2020],
        "population": [10, 20, 30, 40, 50],
    }
)

mock_countries_regions = pd.DataFrame(
    {
        "code": ["C01", "C02", "C03", "R01", "R02"],
        "name": ["Country 1", "Country 2", "Country 3", "Region 1", "Region 2"],
        "members": [np.nan, np.nan, np.nan, '["C01", "C02"]', '["C03"]'],
    }
).set_index("code")

mock_income_groups = pd.DataFrame(
    {
        "country": ["Country 2", "Country 3", "Country 1"],
        "income_group": ["Income group 1", "Income group 1", "Income group 2"],
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

    def test_with_dataframes_of_equal_values_but_different_indexes(self):
        # Even if dataframes are not identical, compare_dataframes should return all Trues (since it does not care about
        # indexes, only values).
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [1, 2], "col_02": ["a", "b"]}).set_index(
                "col_02"
            ),
            df2=pd.DataFrame({"col_01": [1, 2], "col_02": ["a", "c"]}).set_index(
                "col_02"
            ),
        ).equals(pd.DataFrame({"col_01": [True, True]}))

    def test_with_two_dataframes_with_object_columns_with_nans(self):
        assert utils.compare_dataframes(
            df1=pd.DataFrame({"col_01": [np.nan, "b", "c"]}),
            df2=pd.DataFrame({"col_01": [np.nan, "b", "c"]}),
        ).equals(pd.DataFrame({"col_01": [True, True, True]}))


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

    def test_on_equal_dataframes_with_non_numeric_indexes(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2], "col_02": ["a", "b"]}).set_index(
                "col_02"
            ),
            df2=pd.DataFrame({"col_01": [1, 2], "col_02": ["a", "b"]}).set_index(
                "col_02"
            ),
        )[0]

    def test_on_dataframes_of_equal_values_but_different_indexes(self):
        assert not utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [1, 2], "col_02": ["a", "b"]}).set_index(
                "col_02"
            ),
            df2=pd.DataFrame({"col_01": [1, 2], "col_02": ["a", "c"]}).set_index(
                "col_02"
            ),
        )[0]

    def test_on_dataframes_with_object_columns_with_nans(self):
        assert utils.are_dataframes_equal(
            df1=pd.DataFrame({"col_01": [np.nan, "b", "c"]}),
            df2=pd.DataFrame({"col_01": [np.nan, "b", "c"]}),
        )[0]


@patch.object(utils.catalog, "find", MockPopulationLoad)
class TestAddPopulationToDataframe:
    def test_all_countries_and_years_in_population(self):
        df_in = pd.DataFrame(
            {"country": ["Country 2", "Country 1"], "year": [2019, 2021]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country 2", "Country 1"],
                "year": [2019, 2021],
                "population": [30, 20],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_countries_and_years_in_population_just_one(self):
        df_in = pd.DataFrame(
            {"country": ["Country 2", "Country 2"], "year": [2020, 2019]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country 2", "Country 2"],
                "year": [2020, 2019],
                "population": [40, 30],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_one_country_in_and_another_not_in_population(self):
        df_in = pd.DataFrame(
            {"country": ["Country 1", "Country 3"], "year": [2020, 2021]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country 1", "Country 3"],
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
            {"country": ["Country 2", "Country 1"], "year": [2000, 2000]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country 2", "Country 1"],
                "year": [2000, 2000],
                "population": [np.nan, np.nan],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_countries_in_population_but_a_year_in_and_another_not_in_population(self):
        df_in = pd.DataFrame(
            {"country": ["Country 2", "Country 1"], "year": [2019, 2000]}
        )
        df_out = pd.DataFrame(
            {
                "country": ["Country 2", "Country 1"],
                "year": [2019, 2000],
                "population": [30, np.nan],
            }
        )
        assert utils.add_population_to_dataframe(df=df_in).equals(df_out)

    def test_change_country_and_year_column_names(self):
        df_in = pd.DataFrame(
            {"Country": ["Country 2", "Country 1"], "Year": [2019, 2021]}
        )
        df_out = pd.DataFrame(
            {
                "Country": ["Country 2", "Country 1"],
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
            {"country": ["Country 1", "country_02"], "some_variable": [1, 2]}
        )
        df_out = pd.DataFrame(
            {"country": ["Country 1", "Country 2"], "some_variable": [1, 2]}
        )
        assert utils.standardize_countries(df=df_in, countries_file="IGNORED").equals(
            df_out
        )

    def test_one_country_unchanged_and_another_unknown(self, _):
        df_in = pd.DataFrame(
            {"country": ["Country 1", "country_04"], "some_variable": [1, 2]}
        )
        df_out = pd.DataFrame(
            {"country": ["Country 1", "country_04"], "some_variable": [1, 2]}
        )
        assert utils.standardize_countries(df=df_in, countries_file="IGNORED").equals(
            df_out
        )

    def test_two_unknown_countries_made_nan(self, _):
        df_in = pd.DataFrame(
            {"country": ["Country 1", "country_04"], "some_variable": [1, 2]}
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
            {"country": ["Country 1", "country_02"], "some_variable": [1, 2]}
        )
        df_out = pd.DataFrame(
            {"country": [np.nan, "Country 2"], "some_variable": [1, 2]}
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
                "country": ["Country 1", "country_02", "Country 3"],
                "some_variable": [1, 2, 3],
            }
        )
        df_out = pd.DataFrame(
            {
                "country": [np.nan, "Country 2", "Country 3"],
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
        df_out = pd.DataFrame({"Country": ["Country 2"]})
        assert utils.are_dataframes_equal(
            df1=df_out,
            df2=utils.standardize_countries(
                df=df_in, countries_file="IGNORED", country_col="Country"
            ),
        )


class TestGroupbyAggregate:
    def test_default_aggregate_single_groupby_column_as_string(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2003, 2003, 2003, 2002, 2002],
                "value_01": [1, 2, 3, 4, 5, 6],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [1, 11, 9],
            }
        ).set_index("year")
        assert utils.groupby_agg(
            df_in,
            "year",
            aggregations=None,
            num_allowed_nans=None,
            frac_allowed_nans=None,
        ).equals(df_out)

    def test_default_aggregate_single_groupby_column_as_list(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2003, 2003, 2003, 2002, 2002],
                "value_01": [1, 2, 3, 4, 5, 6],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [1, 11, 9],
            }
        ).set_index("year")
        assert utils.groupby_agg(
            df_in,
            ["year"],
            aggregations=None,
            num_allowed_nans=None,
            frac_allowed_nans=None,
        ).equals(df_out)

    def test_default_aggregate_with_some_nans_ignored(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [0.0, 2.0, 15.0],
            }
        ).set_index("year")
        assert utils.groupby_agg(
            df_in,
            ["year"],
            aggregations=None,
            num_allowed_nans=None,
            frac_allowed_nans=None,
        ).equals(df_out)

    def test_default_aggregate_with_some_nans_ignored_different_types(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": ["a", "b", "c", "d", "e", "f"],
                "value_03": [True, False, False, True, True, False],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [0.0, 2.0, 15.0],
                "value_02": ["a", "bc", "def"],
                "value_03": [1, 0, 2],
            }
        ).set_index("year")
        assert utils.groupby_agg(
            df_in,
            ["year"],
            aggregations=None,
            num_allowed_nans=None,
            frac_allowed_nans=None,
        ).equals(df_out)

    def test_default_aggregate_with_some_nans_ignored_different_types_and_more_nans(
        self,
    ):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f"],
                "value_03": [np.nan, False, False, True, True, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [0.0, 2.0, 15.0],
                "value_02": [0, "b", "def"],
                "value_03": [0, 0, 2],
            }
        ).set_index("year")
        df_out["value_03"] = df_out["value_03"].astype(object)
        assert utils.groupby_agg(
            df_in,
            ["year"],
            aggregations=None,
            num_allowed_nans=None,
            frac_allowed_nans=None,
        ).equals(df_out)

    def test_default_aggregate_with_num_allowed_nans_zero(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f"],
                "value_03": [np.nan, False, False, True, True, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [np.nan, np.nan, 15.0],
                "value_02": [np.nan, np.nan, "def"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [np.nan, 0, np.nan], index=[2001, 2002, 2003], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=0,
                frac_allowed_nans=None,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_num_allowed_nans_one(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f"],
                "value_03": [np.nan, False, False, True, np.nan, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [0.0, 2.0, 15.0],
                "value_02": [0, "b", "def"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [0, 0, np.nan], index=[2001, 2002, 2003], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=1,
                frac_allowed_nans=None,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_num_allowed_nans_two(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f"],
                "value_03": [np.nan, False, False, True, np.nan, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [0.0, 2.0, 15.0],
                "value_02": [0, "b", "def"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [0, 0, 1], index=[2001, 2002, 2003], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=2,
                frac_allowed_nans=None,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_num_allowed_nans_the_length_of_the_dataframe(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2004, 2004, 2004, 2004],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6, 7],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f", "g"],
                "value_03": [np.nan, False, False, True, np.nan, np.nan, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2004],
                "value_01": [0.0, 2.0, 22.0],
                "value_02": [0, "b", "defg"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [0, 0, 1], index=[2001, 2002, 2004], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=None,
                frac_allowed_nans=len(df_in),
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_frac_allowed_nans_zero(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f"],
                "value_03": [np.nan, False, False, True, True, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [np.nan, np.nan, 15.0],
                "value_02": [np.nan, np.nan, "def"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [np.nan, 0, np.nan], index=[2001, 2002, 2003], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=None,
                frac_allowed_nans=0,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_frac_allowed_nans_half(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f"],
                "value_03": [np.nan, False, False, True, np.nan, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [np.nan, 2.0, 15.0],
                "value_02": [np.nan, "b", "def"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [np.nan, 0, np.nan], index=[2001, 2002, 2003], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=None,
                frac_allowed_nans=0.5,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_frac_allowed_nans_two_thirds(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f"],
                "value_03": [np.nan, False, False, True, np.nan, np.nan],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [np.nan, 2.0, 15.0],
                "value_02": [np.nan, "b", "def"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [np.nan, 0, 1], index=[2001, 2002, 2003], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=None,
                frac_allowed_nans=0.67,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_frac_allowed_nans_one(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003, 2004, 2004, 2004, 2004],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6, 7, np.nan, np.nan, np.nan],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f", "g", "h", "i", "j"],
                "value_03": [
                    np.nan,
                    False,
                    False,
                    True,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    True,
                ],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003, 2004],
                "value_01": [0, 2.0, 15.0, 7],
                "value_02": [0, "b", "def", "ghij"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [0, 0, 1, 1], index=[2001, 2002, 2003, 2004], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=None,
                frac_allowed_nans=None,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_both_num_allowed_nans_and_frac_allowed_nans(self):
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003, 2004, 2004, 2004, 2004],
                "value_01": [np.nan, 2, np.nan, 4, 5, 6, 7, np.nan, np.nan, np.nan],
                "value_02": [np.nan, "b", np.nan, "d", "e", "f", "g", "h", "i", "j"],
                "value_03": [
                    np.nan,
                    False,
                    False,
                    True,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    True,
                ],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003, 2004],
                "value_01": [np.nan, 2.0, 15.0, np.nan],
                "value_02": [np.nan, "b", "def", "ghij"],
            }
        ).set_index("year")
        df_out["value_03"] = pd.Series(
            [np.nan, 0, np.nan, np.nan], index=[2001, 2002, 2003, 2004], dtype=object
        )
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=None,
                num_allowed_nans=2,
                frac_allowed_nans=0.5,
            ),
            df2=df_out,
        )[0]

    def test_default_aggregate_with_two_groupby_columns(self):
        df_in = pd.DataFrame(
            {
                "country": [
                    "country_a",
                    "country_a",
                    "country_a",
                    "country_b",
                    "country_b",
                    "country_c",
                ],
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [1, 2, 3, 4, 5, 6],
            }
        )
        df_out = pd.DataFrame(
            {
                "country": ["country_a", "country_a", "country_b", "country_c"],
                "year": [2001, 2002, 2003, 2003],
                "value_01": [1, 5, 9, 6],
            }
        ).set_index(["country", "year"])
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["country", "year"],
                aggregations=None,
                num_allowed_nans=None,
                frac_allowed_nans=None,
            ),
            df2=df_out,
        )[0]

    def test_custom_aggregate(self):
        aggregations = {"value_01": "sum", "value_02": "mean"}
        df_in = pd.DataFrame(
            {
                "year": [2001, 2002, 2002, 2003, 2003, 2003],
                "value_01": [1, 2, 3, 4, 5, np.nan],
                "value_02": [1, 2, 3, 4, 5, 6],
            }
        )
        df_out = pd.DataFrame(
            {
                "year": [2001, 2002, 2003],
                "value_01": [1.0, 5.0, np.nan],
                "value_02": [1, 2.5, 7.5],
            }
        ).set_index("year")
        assert utils.are_dataframes_equal(
            df1=utils.groupby_agg(
                df_in,
                ["year"],
                aggregations=aggregations,
                num_allowed_nans=0,
                frac_allowed_nans=None,
            ),
            df2=df_out,
        )


class TestMultiMerge:
    df1 = pd.DataFrame({"col_01": ["aa", "ab", "ac"], "col_02": ["ba", "bb", "bc"]})

    def test_merge_identical_dataframes(self):
        df1 = self.df1.copy()
        df2 = self.df1.copy()
        df3 = self.df1.copy()
        assert utils.multi_merge(
            [df1, df2, df3], how="inner", on=["col_01", "col_02"]
        ).equals(df1)

    def test_inner_join_with_non_overlapping_dataframes(self):
        df1 = self.df1.copy()
        df2 = pd.DataFrame({"col_01": ["ad", "ae"]})
        df3 = pd.DataFrame({"col_01": ["af"], "col_03": ["ca"]})
        # For some reason the order of columns changes on the second merge.
        df_out = pd.DataFrame({"col_02": [], "col_01": [], "col_03": []}, dtype=str)
        assert utils.are_dataframes_equal(
            df1=utils.multi_merge([df1, df2, df3], how="inner", on="col_01"), df2=df_out
        )

    def test_outer_join_with_non_overlapping_dataframes(self):
        df1 = self.df1.copy()
        df2 = pd.DataFrame({"col_01": ["ad"]})
        df3 = pd.DataFrame({"col_01": ["ae"]})
        df_out = pd.DataFrame(
            {
                "col_01": ["aa", "ab", "ac", "ad", "ae"],
                "col_02": ["ba", "bb", "bc", np.nan, np.nan],
            }
        )
        assert utils.are_dataframes_equal(
            df1=utils.multi_merge([df1, df2, df3], how="outer", on="col_01"), df2=df_out
        )[0]

    def test_left_join(self):
        df1 = self.df1.copy()
        df2 = pd.DataFrame(
            {
                "col_01": ["aa", "ab", "ad"],
                "col_02": ["ba", "bB", "bc"],
                "col_03": [1, 2, 3],
            }
        )
        # df_12 = pd.DataFrame({'col_01': ['aa', 'ab', 'ac'], 'col_02': ['ba', 'bb', 'bc'],
        #                       'col_03': [1, np.nan, np.nan]})
        df3 = pd.DataFrame({"col_01": [], "col_02": [], "col_04": []})
        df_out = pd.DataFrame(
            {
                "col_01": ["aa", "ab", "ac"],
                "col_02": ["ba", "bb", "bc"],
                "col_03": [1, np.nan, np.nan],
                "col_04": [np.nan, np.nan, np.nan],
            }
        )
        assert utils.multi_merge(
            [df1, df2, df3], how="left", on=["col_01", "col_02"]
        ).equals(df_out)

    def test_right_join(self):
        df1 = self.df1.copy()
        df2 = pd.DataFrame(
            {
                "col_01": ["aa", "ab", "ad"],
                "col_02": ["ba", "bB", "bc"],
                "col_03": [1, 2, 3],
            }
        )
        # df12 = pd.DataFrame({'col_01': ['aa', 'ab', 'ad'], 'col_02': ['ba', 'bB', 'bc'], 'col_03': [1, 2, 3]})
        df3 = pd.DataFrame(
            {"col_01": ["aa", "ae"], "col_02": ["ba", "be"], "col_04": [4, 5]}
        )
        df_out = pd.DataFrame(
            {
                "col_01": ["aa", "ae"],
                "col_02": ["ba", "be"],
                "col_03": [1, np.nan],
                "col_04": [4, 5],
            }
        )
        assert utils.multi_merge(
            [df1, df2, df3], how="right", on=["col_01", "col_02"]
        ).equals(df_out)


class TestListCountriesInRegions(unittest.TestCase):
    def test_get_countries_from_region(self):
        assert utils.list_countries_in_region(
            region="Region 1",
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
        ) == ["Country 1", "Country 2"]

    def test_get_countries_from_another_region(self):
        assert utils.list_countries_in_region(
            region="Region 2",
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
        ) == ["Country 3"]

    def test_get_countries_from_income_group(self):
        assert utils.list_countries_in_region(
            region="Income group 1",
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
        ) == ["Country 2", "Country 3"]

    def test_get_countries_from_another_income_group(self):
        assert utils.list_countries_in_region(
            region="Income group 2",
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
        ) == ["Country 1"]

    def test_raise_error_for_unknown_region(self):
        with self.assertRaises(utils.RegionNotFound):
            utils.list_countries_in_region(
                region="Made-up region",
                countries_regions=mock_countries_regions,
                income_groups=mock_income_groups,
            )

    def test_empty_region(self):
        assert (
            utils.list_countries_in_region(
                region="Country 1",
                countries_regions=mock_countries_regions,
                income_groups=mock_income_groups,
            )
            == []
        )


class TestListCountriesInRegionsThatMustHaveData(unittest.TestCase):
    def test_having_too_loose_conditions(self):
        with self.assertWarns(UserWarning):
            assert (
                utils.list_countries_in_region_that_must_have_data(
                    region="Region 1",
                    reference_year=2020,
                    min_frac_individual_population=0.0,
                    min_frac_cumulative_population=0.0,
                    countries_regions=mock_countries_regions,
                    income_groups=mock_income_groups,
                    population=mock_population,
                )
                == []
            )

    def test_having_too_strict_condition_on_minimum_individual_contribution(self):
        with self.assertWarns(UserWarning):
            assert utils.list_countries_in_region_that_must_have_data(
                region="Region 1",
                reference_year=2020,
                min_frac_individual_population=0.81,
                min_frac_cumulative_population=0.0,
                countries_regions=mock_countries_regions,
                income_groups=mock_income_groups,
                population=mock_population,
            ) == ["Country 2", "Country 1"]

    def test_having_too_strict_condition_on_minimum_cumulative_contribution(self):
        with self.assertWarns(UserWarning):
            assert utils.list_countries_in_region_that_must_have_data(
                region="Region 1",
                reference_year=2020,
                min_frac_individual_population=0.0,
                min_frac_cumulative_population=0.81,
                countries_regions=mock_countries_regions,
                income_groups=mock_income_groups,
                population=mock_population,
            ) == ["Country 2", "Country 1"]

    def test_having_too_strict_condition_on_both_minimum_individual_and_cumulative_contributions(
        self,
    ):
        with self.assertWarns(UserWarning):
            assert utils.list_countries_in_region_that_must_have_data(
                region="Region 1",
                reference_year=2020,
                min_frac_individual_population=0.81,
                min_frac_cumulative_population=0.81,
                countries_regions=mock_countries_regions,
                income_groups=mock_income_groups,
                population=mock_population,
            ) == ["Country 2", "Country 1"]

    def test_region_year_with_only_one_country(self):
        assert utils.list_countries_in_region_that_must_have_data(
            region="Region 1",
            reference_year=2021,
            min_frac_individual_population=0.1,
            min_frac_cumulative_population=0,
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
            population=mock_population,
        ) == ["Country 1"]

    def test_region_year_right_below_minimum_individual_contribution(self):
        assert utils.list_countries_in_region_that_must_have_data(
            region="Region 1",
            reference_year=2020,
            min_frac_individual_population=0.79,
            min_frac_cumulative_population=0.0,
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
            population=mock_population,
        ) == ["Country 2"]

    def test_region_year_right_above_minimum_individual_contribution(self):
        assert utils.list_countries_in_region_that_must_have_data(
            region="Region 1",
            reference_year=2020,
            min_frac_individual_population=0.1,
            min_frac_cumulative_population=0.0,
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
            population=mock_population,
        ) == ["Country 2"]

    def test_region_year_right_below_minimum_cumulative_contribution(self):
        assert utils.list_countries_in_region_that_must_have_data(
            region="Region 1",
            reference_year=2020,
            min_frac_individual_population=0.0,
            min_frac_cumulative_population=0.79,
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
            population=mock_population,
        ) == ["Country 2"]

    def test_region_year_right_above_minimum_cumulative_contribution(self):
        assert utils.list_countries_in_region_that_must_have_data(
            region="Region 1",
            reference_year=2020,
            min_frac_individual_population=0.0,
            min_frac_cumulative_population=0.1,
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
            population=mock_population,
        ) == ["Country 2"]

    def test_countries_in_income_group(self):
        assert utils.list_countries_in_region_that_must_have_data(
            region="Income group 1",
            reference_year=2020,
            min_frac_individual_population=0.0,
            min_frac_cumulative_population=0.5,
            countries_regions=mock_countries_regions,
            income_groups=mock_income_groups,
            population=mock_population,
        ) == ["Country 3"]


@patch.object(utils, "_load_countries_regions", lambda: mock_countries_regions)
@patch.object(utils, "_load_income_groups", lambda: mock_income_groups)
class TestAddRegionAggregates:
    df_in = pd.DataFrame(
        {
            "country": [
                "Country 1",
                "Country 1",
                "Country 2",
                "Country 3",
                "Region 1",
                "Income group 1",
            ],
            "year": [2020, 2021, 2020, 2022, 2022, 2022],
            "var_01": [1, 2, 3, np.nan, 5, 6],
            "var_02": [10, 20, 30, 40, 50, 60],
        }
    )

    def test_add_region_with_one_nan_permitted(self):
        df = utils.add_region_aggregates(
            df=self.df_in,
            region="Region 2",
            countries_in_region=["Country 3"],
            countries_that_must_have_data=["Country 3"],
            num_allowed_nans_per_year=None,
            country_col="country",
            year_col="year",
        )
        df_out = pd.DataFrame(
            {
                "country": [
                    "Country 1",
                    "Country 1",
                    "Country 2",
                    "Country 3",
                    "Region 1",
                    "Income group 1",
                    "Region 2",
                ],
                "year": [2020, 2021, 2020, 2022, 2022, 2022, 2022],
                "var_01": [1, 2, 3, np.nan, 5, 6, 0.0],
                "var_02": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 40.0],
            }
        )
        assert utils.are_dataframes_equal(df1=df, df2=df_out)

    def test_add_region_with_one_nan_not_permitted(self):
        df = utils.add_region_aggregates(
            df=self.df_in,
            region="Region 2",
            countries_in_region=["Country 3"],
            countries_that_must_have_data=["Country 3"],
            num_allowed_nans_per_year=0,
            country_col="country",
            year_col="year",
        )
        df_out = pd.DataFrame(
            {
                "country": [
                    "Country 1",
                    "Country 1",
                    "Country 2",
                    "Country 3",
                    "Region 1",
                    "Income group 1",
                    "Region 2",
                ],
                "year": [2020, 2021, 2020, 2022, 2022, 2022, 2022],
                "var_01": [1, 2, 3, np.nan, 5, 6, np.nan],
                "var_02": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 40.0],
            }
        )
        assert utils.are_dataframes_equal(df1=df, df2=df_out)

    def test_add_income_group(self):
        df = utils.add_region_aggregates(
            df=self.df_in,
            region="Income group 2",
            countries_in_region=["Country 1"],
            countries_that_must_have_data=["Country 1"],
            num_allowed_nans_per_year=0,
            country_col="country",
            year_col="year",
        )
        df_out = pd.DataFrame(
            {
                "country": [
                    "Country 1",
                    "Country 1",
                    "Country 2",
                    "Country 3",
                    "Region 1",
                    "Income group 1",
                    "Income group 2",
                    "Income group 2",
                ],
                "year": [2020, 2021, 2020, 2022, 2022, 2022, 2020, 2021],
                "var_01": [1, 2, 3, np.nan, 5, 6, 1, 2],
                "var_02": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0, 20.0],
            }
        )
        assert utils.are_dataframes_equal(df1=df, df2=df_out)

    def test_replace_region_with_one_non_mandatory_country_missing(self):
        # Country 2 does not have data for 2021, however, since it is not a mandatory country, its data will be treated
        # as zero.
        df = utils.add_region_aggregates(
            df=self.df_in,
            region="Region 1",
            countries_in_region=["Country 1", "Country 2"],
            countries_that_must_have_data=["Country 1"],
            num_allowed_nans_per_year=0,
            country_col="country",
            year_col="year",
        )
        df_out = pd.DataFrame(
            {
                "country": [
                    "Country 1",
                    "Country 1",
                    "Country 2",
                    "Country 3",
                    "Income group 1",
                    "Region 1",
                    "Region 1",
                ],
                "year": [2020, 2021, 2020, 2022, 2022, 2020, 2021],
                "var_01": [1, 2, 3, np.nan, 6, 1 + 3, 2],
                "var_02": [10.0, 20.0, 30.0, 40.0, 60.0, 10.0 + 30.0, 20.0],
            }
        )
        assert utils.are_dataframes_equal(df1=df, df2=df_out)[0]

    def test_replace_region_with_one_mandatory_country_missing(self):
        # Country 2 does not have data for 2021, and, given that it is a mandatory country, the aggregation will be nan.
        df = utils.add_region_aggregates(
            df=self.df_in,
            region="Region 1",
            countries_in_region=["Country 1", "Country 2"],
            countries_that_must_have_data=["Country 1", "Country 2"],
            num_allowed_nans_per_year=0,
            country_col="country",
            year_col="year",
        )
        df_out = pd.DataFrame(
            {
                "country": [
                    "Country 1",
                    "Country 1",
                    "Country 2",
                    "Country 3",
                    "Income group 1",
                    "Region 1",
                    "Region 1",
                ],
                "year": [2020, 2021, 2020, 2022, 2022, 2020, 2021],
                "var_01": [1, 2, 3, np.nan, 6, 1 + 3, np.nan],
                "var_02": [10.0, 20.0, 30.0, 40.0, 60.0, 10.0 + 30.0, np.nan],
            }
        )
        assert utils.are_dataframes_equal(df1=df, df2=df_out)[0]

    def test_replace_region_with_custom_aggregations(self):
        # Country 2 does not have data for 2021, and, given that it is a mandatory country, the aggregation will be nan.
        df = utils.add_region_aggregates(
            df=self.df_in,
            region="Region 1",
            countries_in_region=["Country 1", "Country 2"],
            countries_that_must_have_data=["Country 1", "Country 2"],
            num_allowed_nans_per_year=0,
            country_col="country",
            year_col="year",
            aggregations={"var_01": "sum", "var_02": "mean"},
        )
        df_out = pd.DataFrame(
            {
                "country": [
                    "Country 1",
                    "Country 1",
                    "Country 2",
                    "Country 3",
                    "Income group 1",
                    "Region 1",
                    "Region 1",
                ],
                "year": [2020, 2021, 2020, 2022, 2022, 2020, 2021],
                "var_01": [1, 2, 3, np.nan, 6, 1 + 3, np.nan],
                "var_02": [10.0, 20.0, 30.0, 40.0, 60.0, (10.0 + 30.0) / 2, np.nan],
            }
        )
        assert utils.are_dataframes_equal(df1=df, df2=df_out)[0]

    def test_add_income_group_without_specifying_countries_in_region(self):
        df = utils.add_region_aggregates(
            df=self.df_in,
            region="Income group 2",
            countries_in_region=None,
            countries_that_must_have_data=[],
            num_allowed_nans_per_year=0,
            country_col="country",
            year_col="year",
        )
        df_out = pd.DataFrame(
            {
                "country": [
                    "Country 1",
                    "Country 1",
                    "Country 2",
                    "Country 3",
                    "Region 1",
                    "Income group 1",
                    "Income group 2",
                    "Income group 2",
                ],
                "year": [2020, 2021, 2020, 2022, 2022, 2022, 2020, 2021],
                "var_01": [1, 2, 3, np.nan, 5, 6, 1, 2],
                "var_02": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 10.0, 20.0],
            }
        )
        assert utils.are_dataframes_equal(df1=df, df2=df_out)
