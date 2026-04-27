# Data on Energy by *Our World in Data*

Our complete Energy dataset is a collection of key metrics maintained by [*Our World in Data*](https://ourworldindata.org/energy). It is updated regularly and includes data on energy consumption (primary energy, per capita, and growth rates), energy mix, electricity mix and other relevant metrics.

## The complete *Our World in Data* Energy dataset

### 🗂️ Download our complete Energy dataset : [CSV](https://owid-public.owid.io/data/energy/owid-energy-data.csv) | [XLSX](https://owid-public.owid.io/data/energy/owid-energy-data.xlsx) | [JSON](https://owid-public.owid.io/data/energy/owid-energy-data.json)

The CSV and XLSX files follow a format of 1 row per location and year. The JSON version is split by country, with an array of yearly records.

We will continue to publish updated data on energy as it becomes available. Most metrics are published on an annual basis.

A [full codebook](https://github.com/owid/energy-data/blob/master/owid-energy-codebook.csv) is made available, with a description and source for each indicator in the dataset. This codebook is also included as an additional sheet in the XLSX file.

## Our source data and code

The dataset is built upon a number of datasets and processing steps:
- Statistical review of world energy (Energy Institute, EI):
  - [Source data](https://www.energyinst.org/statistical-review)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/energy_institute/2025-06-27/statistical_review_of_world_energy.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/energy_institute/2025-06-27/statistical_review_of_world_energy.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy_institute/2025-06-27/statistical_review_of_world_energy.py)
- International energy data (U.S. Energy Information Administration, EIA):
  - [Source data](https://www.eia.gov/opendata/bulkfiles.php)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/eia/2025-07-08/international_energy_data.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/eia/2025-07-08/energy_consumption.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/eia/2025-07-08/energy_consumption.py)
- Energy from fossil fuels (The Shift Dataportal):
  - [Source data](https://www.theshiftdataportal.org/energy)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/shift/2023-12-12/energy_production_from_fossil_fuels.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/shift/2023-12-12/energy_production_from_fossil_fuels.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/shift/2023-12-12/energy_production_from_fossil_fuels.py)
- Yearly Electricity Data (Ember):
  - [Source data](https://ember-energy.org/data/yearly-electricity-data/)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/ember/2026-04-24/yearly_electricity.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ember/2026-04-24/yearly_electricity.py)
- Energy mix (Our World in Data based on EI's Statistical review of world energy):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2025-06-27/energy_mix.py)
- Fossil fuel production (Our World in Data based on EI's Statistical review of world energy & The Shift Dataportal's Energy from fossil fuels):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2025-06-27/fossil_fuel_production.py)
- Primary energy consumption (Our World in Data based on EI's Statistical review of world energy & EIA's International energy data):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2025-06-27/primary_energy_consumption.py)
- Electricity mix (Our World in Data based on EI's Statistical Review & Ember's Yearly Electricity Data):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2026-04-24/electricity_mix.py)
- Energy dataset (Our World in Data based on all sources above):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy_data/2026-04-24/owid_energy.py)
  - [Exporting code](https://github.com/owid/etl/blob/master/etl/steps/export/github/energy_data/latest/owid_energy.py)
  - [Uploading code](https://github.com/owid/etl/blob/master/etl/steps/export/s3/energy_data/latest/owid_energy.py)

Additionally, to construct region aggregates and indicators per capita and per GDP, we use the following datasets and processing steps:
- Regions (Our World in Data).
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/regions/2023-01-01/regions.py)
- Population (Our World in Data based on [a number of different sources](https://ourworldindata.org/population-sources)).
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/demography/2024-07-15/population.py)
- Income groups (World Bank).
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/wb/2025-07-01/income_groups.py)
- GDP (University of Groningen GGDC's Maddison Project Database, Bolt and van Zanden, 2024).
  - [Source data](https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2023)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/ggdc/2024-04-26/maddison_project_database.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/ggdc/2024-04-26/maddison_project_database.py)
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ggdc/2024-04-26/maddison_project_database.py)

## Changelog

- 2026-04-27:
  - Update Ember's yearly electricity data, after they updated their Global electricity data.
- 2026-01-26:
  - Update Ember's yearly electricity data, after they updated their European electricity data.
  - Fixed a few issues in aggregate regions related to how Ember's data is combined with the Energy Institute's Statistical Review of World Energy.
- 2026-01-09:
  - Updated Ember's yearly electricity data.
  - Fixed a few small issues (related to "Other *" regions coming from the Energy Institute's Statistical Review of World Energy).
- 2025-11-11:
  - Moved dataset to different URL.
  - Various improvements in the codebook format.
- 2025-07-17:
  - Updated the Energy Institute Statistical Review of World Energy.
  - Updated EIA's International energy data.
- 2025-05-30:
  - Updated Ember's yearly electricity data, which includes data for 2024.
- 2024-09-05:
  - Added per capita electricity demand, from Ember's yearly electricity data.
- 2024-08-30:
  - Fixed coal electricity generation for Switzerland, which was missing in the original data, and should be zero instead.
- 2024-06-20:
  - Updated the Energy Institute Statistical Review of World Energy.
  - Fixed issues on electricity data for aggregate regions.
- 2024-05-08:
  - Updated Ember's yearly electricity data, which includes data for 2023.
  - Updated GDP data, now coming from Maddison Project Database 2023.
- 2024-01-24:
  - Improved codebook, to clarify whether indicators refer to electricity generation or primary energy consumption.
  - Improved the calculation of the share of electricity in primary energy. Previously, electricity generation was calculated as a share of input-equivalent primary energy consumption. Now it is calculated as a share of direct primary energy consumption.
- 2023-12-12:
  - Updated Ember's yearly electricity data and EIA's International energy data.
  - Enhanced codebook (improved descriptions, added units, updated sources).
  - Fixed various minor issues.
- 2023-07-07:
  - Replaced BP's data by the new Energy Institute Statistical Review of World Energy 2023.
  - Updated Ember's yearly electricity data.
  - Updated all datasets accordingly.
- 2023-06-01:
  - Updated Ember's yearly electricity data.
  - Renamed countries 'East Timor' and 'Faroe Islands', and added 'Middle East (Ember)'.
  - Population and per capita indicators are now calculated using an updated version of our population dataset.
- 2023-03-01:
  - Updated Ember's yearly electricity data and fixed some minor issues.
- 2022-12-30:
  - Fixed some minor issues with BP's dataset. Regions like "Other North America (BP)" have been removed from the data, since, in the original Statistical Review of World Energy, these regions represented different sets of countries for different indicators.
- 2022-12-16:
  - The column `electricity_share_energy` (electricity as a share of primary energy) was added to the dataset.
  - Fixed some minor inconsistencies in electricity data between Ember and BP, by prioritizing data from Ember.
  - Updated Ember's yearly electricity data.
- 2022-08-09:
  - All inconsistencies due to different definitions of regions among different datasets (especially Europe) have been fixed.
    - Now all regions follow [Our World in Data's definitions](https://ourworldindata.org/world-region-map-definitions).
    - We also include data for regions as defined in the original datasets; for example, `Europe (BP)` corresponds to Europe as defined by BP.
  - All data processing now occurs outside this repository; the code has been migrated to be part of the [etl repository](https://github.com/owid/etl).
  - Indicator `fossil_cons_per_capita` has been renamed `fossil_elec_per_capita` for consistency, since it corresponds to electricity generation.
  - The codebook has been updated following these changes.
- 2022-04-08:
  - Electricity data from Ember was updated (using the Global Electricity Review 2022).
  - Data on greenhouse-gas emissions in electricity generation was added (`greenhouse_gas_emissions`).
  - Data on emissions intensity is now provided for most countries in the world.
- 2022-03-25:
  - Data on net electricity imports and electricity demand was added.
  - BP data was updated (using the Statistical Review of World Energy 2021).
  - Maddison data on GDP was updated (using the Maddison Project Database 2020).
  - EIA data on primary energy consumption was included in the dataset.
  - Some issues in the dataset were corrected (for example some missing data in production by fossil fuels).
- 2022-02-14:
  - Some issues were corrected in the electricity data, and the energy dataset was updated accordingly.
  - The json and xlsx dataset files were removed from GitHub in favor of an external storage service, to keep this repository at a reasonable size.
  - The `carbon_intensity_elec` column was added back into the energy dataset.
- 2022-02-03:
  - We updated the [Ember global electricity data](https://ember-climate.org/data/global-electricity/), combined with the [European Electricity Review from Ember](https://ember-climate.org/project/european-electricity-review-2022/).
  - The `carbon_intensity_elec` column was removed from the energy dataset (since no updated data was available).
  - Columns for electricity from other renewable sources excluding bioenergy were added (namely `other_renewables_elec_per_capita_exc_biofuel`, and `other_renewables_share_elec_exc_biofuel`).
  - Certain countries and regions have been removed from the dataset, because we identified significant inconsistencies in the original data.
- 2021-03-31:
  - We updated 2020 electricity mix data.
- 2020-09-09:
  - The first version of this dataset was made available.

## Data processing

- **We standardize names of countries and regions.** Since the names of countries and regions are different in different data sources, we standardize all names in order to minimize data loss during data merges.
- **We create aggregate data for regions (e.g. Africa, Europe, etc.).** Since regions are defined differently by our sources, we create our own aggregates following [*Our World in Data* region definitions](https://ourworldindata.org/world-region-map-definitions).
  - We also include data for regions as defined in the original datasets; for example, `Europe (EI)` corresponds to Europe as defined by the Energy Institute.
- **We recalculate primary energy in terawatt-hours.** The primary data sources on energy—the Energy Institute Statistical review of world energy, for example—typically report consumption in terms of exajoules. We have recalculated these figures as terawatt-hours using a conversion factor of 277.8.
  - Primary energy for renewable sources is reported using [the 'substitution method'](https://ourworldindata.org/energy-substitution-method).
- **We calculate per capita figures.** All of our per capita figures are calculated from our `population` metric, which is included in the complete dataset.
  - We also calculate energy consumption per gdp, and include the corresponding `gdp` metric used in the calculation as part of the dataset.
- **We remove inconsistent data.** Certain data points have been removed because their original data presented anomalies. They may be included again in further data releases if the anomalies are amended.

## License

All visualizations, data, and code produced by _Our World in Data_ are completely open access under the [Creative Commons BY license](https://creativecommons.org/licenses/by/4.0/). You have the permission to use, distribute, and reproduce these in any medium, provided the source and authors are credited.

The data produced by third parties and made available by _Our World in Data_ is subject to the license terms from the original third-party authors. We will always indicate the original source of the data in our database, and you should always check the license of any such third-party data before use.

## Authors

This data has been collected, aggregated, and documented by Pablo Rosado, Hannah Ritchie, Edouard Mathieu, and Max Roser.

The mission of *Our World in Data* is to make data and research on the world's largest problems understandable and accessible. [Read more about our mission](https://ourworldindata.org/about).

## How to cite this data?

If you are using this dataset, please cite both [Our World in Data](https://ourworldindata.org/energy#article-citation) and the underlying data source(s).

Please follow [the guidelines in our FAQ](https://ourworldindata.org/faqs#how-should-i-cite-your-data) on how to cite our work.
