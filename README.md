# Data on Energy by *Our World in Data*

Our complete Energy dataset is a collection of key metrics maintained by [*Our World in Data*](https://ourworldindata.org/energy). It is updated regularly and includes data on energy consumption (primary energy, per capita, and growth rates), energy mix, electricity mix and other relevant metrics.

## The complete *Our World in Data* Energy dataset

### 🗂️ Download our complete Energy dataset : [CSV](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv) | [XLSX](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.xlsx) | [JSON](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.json)

The CSV and XLSX files follow a format of 1 row per location and year. The JSON version is split by country, with an array of yearly records.

The variables represent all of our main data related to energy consumption, energy mix, electricity mix as well as other variables of potential interest.

We will continue to publish updated data on energy as it becomes available. Most metrics are published on an annual basis.

A [full codebook](https://github.com/owid/energy-data/blob/master/owid-energy-codebook.csv) is made available, with a description and source for each variable in the dataset.

## Our source data and code

The dataset is built upon a number of datasets and processing steps:
- Statistical review of world energy (BP):
  - [Source data](https://www.bp.com/en/global/corporate/energy-economics/statistical-review-of-world-energy.html)
  - [Ingestion and processing code](https://github.com/owid/importers/tree/master/bp_statreview)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/bp/2022-07-14/statistical_review.py)
- International energy data (EIA):
  - [Source data](https://www.eia.gov/opendata/bulkfiles.php)
  - [Ingestion code](https://github.com/owid/walden/blob/master/ingests/eia_international_energy_data.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/eia/2022-07-27/energy_consumption.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/eia/2022-07-27/energy_consumption.py)
- Energy from fossil fuels (The Shift Dataportal):
  - [Source data](https://www.theshiftdataportal.org/energy)
  - [Ingestion code](https://github.com/owid/walden/blob/master/ingests/shift.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/shift/2022-07-18/fossil_fuel_production.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/shift/2022-07-18/fossil_fuel_production.py)
- Global Electricity Review (Ember):
  - [Source data](https://ember-climate.org/data-catalogue/yearly-electricity-data/)
  - [Ingestion code](https://github.com/owid/walden/blob/master/owid/walden/index/ember/2022-07-25/global_electricity_review.json)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/ember/2022-08-01/global_electricity_review.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ember/2022-08-01/global_electricity_review.py)
- European Electricity Review (Ember):
  - [Source data](https://ember-climate.org/insights/research/european-electricity-review-2022/)
  - [Ingestion code](https://github.com/owid/walden/blob/master/owid/walden/index/ember/2022-02-01/european_electricity_review.json)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/ember/2022-08-01/european_electricity_review.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ember/2022-08-01/european_electricity_review.py)
- Combined Electricity Review (Our World in Data based on Ember's Global and European Electricity Reviews):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ember/2022-08-01/combined_electricity_review.py)
- Energy mix (Our World in Data based on BP's Statistical review of world energy):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/bp/2022-07-14/energy_mix.py)
- Fossil fuel production (Our World in Data based on BP's Statistical review of world energy & Shift's Energy from fossil fuels):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2022-07-20/fossil_fuel_production.py)
- Primary energy consumption (Our World in Data based on BP's Statistical review of world energy & EIA's International energy data):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2022-07-29/primary_energy_consumption.py)
- Electricity mix (Our World in Data based on BP's Statistical Review & Ember's Global and European Electricity Reviews):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2022-08-03/electricity_mix.py)
- Energy dataset (Our World in Data based on all sources above):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2022-08-05/owid_energy.py)
  - [Exporting code](https://github.com/owid/energy-data/blob/master/scripts/make_dataset.py)
  - [Uploading code](https://github.com/owid/energy-data/blob/master/scripts/upload_datasets_to_s3.py)

Additionally, to construct variables per capita and per GDP, we use the following datasets and processing steps:
- Population (Our World in Data based on [a number of different sources](https://ourworldindata.org/population-sources)).
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/owid/latest/key_indicators/table_population.py)
- GDP (University of Groningen GGDC's Maddison Project Database, Bolt and van Zanden, 2020).
  - [Source data](https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2020)
  - [Ingestion code](https://github.com/owid/walden/blob/master/ingests/ggdc_maddison.py)
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ggdc/2020-10-01/ggdc_maddison.py)

## Changelog

- On August 9, 2022:
  - All inconsistencies due to different definitions of regions among different datasets (especially Europe) have been fixed.
    - Now all regions follow [Our World in Data's definitions](https://ourworldindata.org/world-region-map-definitions).
    - We also include data for regions as defined in the original datasets; for example, `Europe (BP)` corresponds to Europe as defined by BP.
  - All data processing now occurs outside this repository; the code has been migrated to be part of the [etl repository](https://github.com/owid/etl).
  - Variable `fossil_cons_per_capita` has been renamed `fossil_elec_per_capita` for consistency, since it corresponds to electricity generation.
  - The codebook has been updated following these changes.
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

- **We standardize names of countries and regions.** Since the names of countries and regions are different in different data sources, we harmonize all names to the [*Our World in Data* standard entity names](https://ourworldindata.org/world-region-map-definitions).
- **We create aggregate data for regions (e.g. Africa, Europe, etc.).** Since regions are defined differently by our sources, we create our own aggregates following [*Our World in Data* region definitions](https://ourworldindata.org/world-region-map-definitions).
  - We also include data for regions as defined in the original datasets; for example, `Europe (BP)` corresponds to Europe as defined by BP.
- **We recalculate primary energy in terawatt-hours.** The primary data sources on energy—the BP Statistical review of world energy, for example—typically report consumption in terms of exajoules. We have recalculated these figures as terawatt-hours using a conversion factor of 277.8.
  - Primary energy for renewable sources is calculated using [the 'substitution method'](https://ourworldindata.org/energy-substitution-method).
- **We calculate per capita figures.** All of our per capita figures are calculated from our `population` metric, which is included in the complete dataset.
  - We also calculate energy consumption per gdp, and include the corresponding `gdp` metric used in the calculation as part of the dataset.
- **We remove inconsistent data.** Certain data points have been removed because their original data presented anomalies. They may be included again in further data releases if the anomalies are amended.

## License

All visualizations, data, and code produced by _Our World in Data_ are completely open access under the [Creative Commons BY license](https://creativecommons.org/licenses/by/4.0/). You have the permission to use, distribute, and reproduce these in any medium, provided the source and authors are credited.

The data produced by third parties and made available by _Our World in Data_ is subject to the license terms from the original third-party authors. We will always indicate the original source of the data in our database, and you should always check the license of any such third-party data before use.

## Authors

This data has been collected, aggregated, and documented by Hannah Ritchie, Pablo Rosado, Edouard Mathieu, Max Roser.

*Our World in Data* makes data and research on the world’s largest problems understandable and accessible. [Read more about our mission](https://ourworldindata.org/about).
