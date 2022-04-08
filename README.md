# Data on Energy by *Our World in Data*

Our complete Energy dataset is a collection of key metrics maintained by [*Our World in Data*](https://ourworldindata.org/energy). It is updated regularly and includes data on energy consumption (primary energy, per capita, and growth rates), energy mix, electricity mix and other relevant metrics.

### 🗂️ Download our complete Energy dataset : [CSV](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv) | [XLSX](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.xlsx) | [JSON](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.json)

We will continue to publish updated data on Energy as it becomes available. Most metrics are published on an annual basis.

## Our data sources

- **Energy consumption (primary energy, energy mix and energy intensity):** this data is sourced from a combination of three sources—the [BP Statistical Review of World Energy](https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy.html), the [EIA](https://www.eia.gov/international/data/world/total-energy/more-total-energy-data) and the [SHIFT Data Portal](https://www.theshiftdataportal.org/energy).
- **Electricity generation (electricity generation, and electricity mix):** this data is sourced from a combination of three sources—the [BP Statistical Review of World Energy](https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy.html), the [Ember – Data Explorer](https://ember-climate.org/global-electricity-review-2021/data-explorer/) and the [Ember European Electricity Review](https://ember-climate.org/project/european-electricity-review-2022/).
- **Other variables:** this data is collected from a variety of sources (United Nations, World Bank, Gapminder, Maddison Project Database, etc.). More information is available in [our codebook](https://github.com/owid/energy-data/blob/master/owid-energy-codebook.csv).

## The complete *Our World in Data* Energy dataset

**Our complete Energy dataset is available in [CSV](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv), [XLSX](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.xlsx), and [JSON](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.json) formats.**

The CSV and XLSX files follow a format of 1 row per location and year. The JSON version is split by country, with an array of yearly records.

The variables represent all of our main data related to energy consumption, energy mix, electricity mix as well as other variables of potential interest.

A [full codebook](https://github.com/owid/energy-data/blob/master/owid-energy-codebook.csv) is made available, with a description and source for each variable in the dataset.

## Changelog

- On April 8, 2022:
  - Electricity data from Ember was updated (using the Global Electricity Review 2022).
  - Data on greenhouse-gas emissions in electricity generation was added (`greenhouse_gas_emissions`).
  - Data on emissions intensity is now provided for most countries in the world.
- On March 25, 2022:
  - Data on net electricity imports and electricity demand was added.
  - BP data was updated (using the Statistical Review of the World Energy 2021).
  - Maddison data on GDP was updated (using the Maddison Project Database 2020).
  - EIA data on primary energy consumption was included in the dataset.
  - Some issues in the dataset were corrected (for example some missing data in production by fossil fuels).
- On February 14, 2022:
  - Some issues were corrected in the electricity data, and the energy dataset was updated accordingly.
  - The json and xlsx dataset files were removed from GitHub in favor of an external storage service, to keep this repository at a reasonable size.
  - The `carbon_intensity_elec` column was added back into the energy dataset.
- On February 3, 2022, we updated the [Ember global electricity data](https://ember-climate.org/data/global-electricity/), combined with the [European Electricity Review from Ember](https://ember-climate.org/project/european-electricity-review-2022/).
  - The `carbon_intensity_elec` column was removed from the energy dataset (since no updated data was available).
  - Columns for electricity from other renewable sources excluding bioenergy were added (namely `other_renewables_elec_per_capita_exc_biofuel`, and `other_renewables_share_elec_exc_biofuel`).
  - Certain countries and regions have been removed from the dataset, because we identified significant inconsistencies in the original data.
- On March 31, 2021, we updated 2020 electricity mix data.
- On September 9, 2020, the first version of this dataset was made available.

## Data alterations

- **We standardize names of countries and regions.** Since the names of countries and regions are different in different data sources, we standardize all names to the [*Our World in Data* standard entity names](https://github.com/owid/energy-data/tree/master/scripts/input/shared).
- **We recalculate primary energy in terawatt-hours.** The primary data sources on energy—the BP Statistical Review of World Energy, for example—typically report consumption in terms of exajoules. We have recalculated these figures as terawatt-hours using a conversion factor of 277.8.
- **We calculate per capita figures.** All of our per capita figures are calculated from our metric `Population`, which is included in the complete dataset. These population figures are sourced from [Gapminder](http://gapminder.org) and the [UN World Population Prospects (UNWPP)](https://population.un.org/wpp/).
- **We remove inconsistent data.** Certain countries and regions have been removed because their original data presented anomalies. They may be included again in further data releases.

## License

All visualizations, data, and code produced by _Our World in Data_ are completely open access under the [Creative Commons BY license](https://creativecommons.org/licenses/by/4.0/). You have the permission to use, distribute, and reproduce these in any medium, provided the source and authors are credited.

The data produced by third parties and made available by _Our World in Data_ is subject to the license terms from the original third-party authors. We will always indicate the original source of the data in our database, and you should always check the license of any such third-party data before use.

## Authors

This data has been collected, aggregated, and documented by Hannah Ritchie, Pablo Rosado, Edouard Mathieu, Max Roser.

The mission of *Our World in Data* is to make data and research on the world’s largest problems understandable and accessible. [Read more about our mission](https://ourworldindata.org/about).
