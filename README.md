# Data on Energy by *Our World in Data*

Our complete Energy dataset is a collection of key metrics maintained by [*Our World in Data*](https://ourworldindata.org/energy). It is updated regularly and includes data on energy consumption (primary energy, per capita, and growth rates), energy mix, electricity mix and other relevant metrics.

## The complete *Our World in Data* Energy dataset

### 🗂️ Download our complete Energy dataset : [CSV](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv) | [XLSX](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.xlsx) | [JSON](https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.json)

The CSV and XLSX files follow a format of 1 row per location and year. The JSON version is split by country, with an array of yearly records.

The indicators represent all of our main data related to energy consumption, energy mix, electricity mix as well as other indicators of potential interest.

We will continue to publish updated data on energy as it becomes available. Most metrics are published on an annual basis.

A [full codebook](https://github.com/owid/energy-data/blob/master/owid-energy-codebook.csv) is made available, with a description and source for each indicator in the dataset. This codebook is also included as an additional sheet in the XLSX file.

## Our source data and code

The dataset is built upon a number of datasets and processing steps:
- Statistical review of world energy (Energy Institute, EI):
  - [Source data](https://www.energyinst.org/statistical-review)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/energy_institute/2024-06-20/statistical_review_of_world_energy.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/energy_institute/2024-06-20/statistical_review_of_world_energy.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy_institute/2024-06-20/statistical_review_of_world_energy.py)
- International energy data (U.S. Energy Information Administration, EIA):
  - [Source data](https://www.eia.gov/opendata/bulkfiles.php)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/eia/2023-12-12/international_energy_data.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/eia/2023-12-12/energy_consumption.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/eia/2023-12-12/energy_consumption.py)
- Energy from fossil fuels (The Shift Dataportal):
  - [Source data](https://www.theshiftdataportal.org/energy)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/shift/2023-12-12/energy_production_from_fossil_fuels.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/shift/2023-12-12/energy_production_from_fossil_fuels.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/shift/2023-12-12/energy_production_from_fossil_fuels.py)
- Yearly Electricity Data (Ember):
  - [Source data](https://ember-climate.org/data-catalogue/yearly-electricity-data/)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/ember/2024-05-08/yearly_electricity.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/ember/2024-05-08/yearly_electricity.py)
  - [Further processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ember/2024-05-08/yearly_electricity.py)
- Energy mix (Our World in Data based on EI's Statistical review of world energy):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2024-06-20/energy_mix.py)
- Fossil fuel production (Our World in Data based on EI's Statistical review of world energy & The Shift Dataportal's Energy from fossil fuels):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2024-06-20/fossil_fuel_production.py)
- Primary energy consumption (Our World in Data based on EI's Statistical review of world energy & EIA's International energy data):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2024-06-20/primary_energy_consumption.py)
- Electricity mix (Our World in Data based on EI's Statistical Review & Ember's Yearly Electricity Data):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/energy/2024-06-20/electricity_mix.py)
- Energy dataset (Our World in Data based on all sources above):
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/external/energy_data/latest/owid_energy.py)
  - [Exporting code](https://github.com/owid/energy-data/blob/master/scripts/make_dataset.py)
  - [Uploading code](https://github.com/owid/energy-data/blob/master/scripts/upload_datasets_to_s3.py)

Additionally, to construct region aggregates and indicators per capita and per GDP, we use the following datasets and processing steps:
- Regions (Our World in Data).
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/regions/2023-01-01/regions.py)
- Population (Our World in Data based on [a number of different sources](https://ourworldindata.org/population-sources)).
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/demography/2023-03-31/population/__init__.py)
- Income groups (World Bank).
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/wb/2024-03-11/income_groups.py)
- GDP (University of Groningen GGDC's Maddison Project Database, Bolt and van Zanden, 2024).
  - [Source data](https://www.rug.nl/ggdc/historicaldevelopment/maddison/releases/maddison-project-database-2023)
  - [Ingestion code](https://github.com/owid/etl/blob/master/snapshots/ggdc/2024-04-26/maddison_project_database.py)
  - [Basic processing code](https://github.com/owid/etl/blob/master/etl/steps/data/meadow/ggdc/2024-04-26/maddison_project_database.py)
  - [Processing code](https://github.com/owid/etl/blob/master/etl/steps/data/garden/ggdc/2024-04-26/maddison_project_database.py)

## Changelog

- On September 5, 2024:
  - Added per capita electricity demand, from Ember's yearly electricity data.
- On August 30, 2024:
  - Fixed coal electricity generation for Switzerland, which was missing in the original data, and should be zero instead.
- On June 20, 2024:
  - Updated the Energy Institute Statistical Review of World Energy.
  - Fixed issues on electricity data for aggregate regions.
- On May 8, 2024:
  - Updated Ember's yearly electricity data, which includes data for 2023.
  - Updated GDP data, now coming from Maddison Project Database 2023.
- On January 24, 2024:
  - Improved codebook, to clarify whether indicators refer to electricity generation or primary energy consumption.
  - Improved the calculation of the share of electricity in primary energy. Previously, electricity generation was calculated as a share of input-equivalent primary energy consumption. Now it is calculated as a share of direct primary energy consumption.
- On December 12, 2023:
  - Updated Ember's yearly electricity data and EIA's International energy data.
  - Enhanced codebook (improved descriptions, added units, updated sources).
  - Fixed various minor issues.
- On July 7, 2023:
  - Replaced BP's data by the new Energy Institute Statistical Review of World Energy 2023.
  - Updated Ember's yearly electricity data.
  - Updated all datasets accordingly.
- On June 1, 2023:
  - Updated Ember's yearly electricity data.
  - Renamed countries 'East Timor' and 'Faroe Islands', and added 'Middle East (Ember)'.
  - Population and per capita indicators are now calculated using an updated version of our population dataset.
- On March 1, 2023:
  - Updated Ember's yearly electricity data and fixed some minor issues.
- On December 30, 2022:
  - Fixed some minor issues with BP's dataset. Regions like "Other North America (BP)" have been removed from the data, since, in the original Statistical Review of World Energy, these regions represented different sets of countries for different indicators.
- On December 16, 2022:
  - The column `electricity_share_energy` (electricity as a share of primary energy) was added to the dataset.
  - Fixed some minor inconsistencies in electricity data between Ember and BP, by prioritizing data from Ember.
  - Updated Ember's yearly electricity data.
- On August 9, 2022:
  - All inconsistencies due to different definitions of regions among different datasets (especially Europe) have been fixed.
    - Now all regions follow [Our World in Data's definitions](https://ourworldindata.org/world-region-map-definitions).
    - We also include data for regions as defined in the original datasets; for example, `Europe (BP)` corresponds to Europe as defined by BP.
  - All data processing now occurs outside this repository; the code has been migrated to be part of the [etl repository](https://github.com/owid/etl).
  - Indicator `fossil_cons_per_capita` has been renamed `fossil_elec_per_capita` for consistency, since it corresponds to electricity generation.
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

This data has been collected, aggregated, and documented by Hannah Ritchie, Pablo Rosado, Edouard Mathieu, Max Roser.

*Our World in Data* makes data and research on the world’s largest problems understandable and accessible. [Read more about our mission](https://ourworldindata.org/about).

## How to cite this data?

If you are using this dataset, please cite both [Our World in Data](https://ourworldindata.org/energy#citation) and the underlying data source(s).

Please follow [the guidelines in our FAQ](https://ourworldindata.org/faqs#citing-work-produced-by-third-parties-and-made-available-by-our-world-in-data) on how to cite our work.
