# ng-growth-poverty

An analysis of whether economic growth actually reduces poverty - and whether it
does so equally across different African countries. The short answer is: growth
helps, but not equally, and Nigeria gets less poverty reduction per unit of growth
than its peers.

---

## What This Project Does

This project looks at five African countries - Nigeria, Ghana, Kenya, South Africa,
and Ethiopia and asks a simple but important question: 
**When a country's economy grows, do the people at the bottom actually benefit?**

To answer this, we use 34 years of World Bank data (1990-2023) and three layers
of analysis:

1. **Nigeria in depth** - Is the relationship between GDP and poverty real and
   stable or does it break down during shocks like the 2016 oil crash or COVID?
2. **Across all five countries** - Does a unit of economic growth buy the same
   amount of poverty reduction in Nigeria as it does in Ghana or Ethiopia?
3. **What else matters** - Beyond GDP, do inequality, inflation or foreign aid
   independently affect poverty?

---

## What We Found

| Question | Answer |
|---|---|
| Does growth reduce poverty in Nigeria? | Yes. A sustained increase in GDP per capita is reliably associated with falling poverty and this held even through major economic shocks. |
| Did the 2016 recession or COVID break the relationship? | No. The underlying connection between growth and poverty stayed intact. The shocks caused temporary setbacks not a permanent change in how growth transmits to welfare. |
| Does growth reduce poverty equally across countries? | No. Ghana and South Africa get significantly more poverty reduction per unit of growth than Nigeria. Kenya is the only country where poverty actually rose as GDP grew - still unexplained. |
| What drives poverty beyond GDP? | Nothing significant. Once we account for each country's fixed characteristics, only GDP matters. Inequality, inflation, and foreign aid do not independently predict poverty changes within countries. |

**The headline finding:** Nigeria's economy grew substantially between 2000 and 2015 -
GDP per capita nearly doubled yet poverty only fell by about 19%.
Over the same period, Ethiopia achieved a 40% poverty drop from a
much lower starting point. Growth in Nigeria happened but the gains concentrated
in oil and related sectors that employ relatively few people and do not raise wages
at the bottom of the income scale.

---

## The Data

All data comes from the **World Bank Open Data** API, pulled automatically via the
`wbgapi` Python package. No manual download is needed.

| Indicator | What it measures | Data quality |
|---|---|---|
| GDP per capita | Income level per person | Clean, annual, all countries |
| GDP growth rate | How fast the economy is growing | Clean, annual, all countries |
| Poverty headcount | Share of population living below $2.15/day | Sparse - surveys happen every 2-7 years |
| Gini index | How unequally income is distributed | Sparse - same survey limitation |
| Net ODA | Foreign aid received | Clean, annual, all countries |
| Inflation | Rate of price increases | Clean, annual, all countries |
| Unemployment | Share of workforce without jobs | Clean, annual, all countries |
| Trade % of GDP | How open the economy is to trade | Missing entirely for Nigeria |

The poverty and Gini data are the trickiest part. Because household surveys only
happen every few years, we have just 5-8 data points per country over 34 years.
We fill the gaps using linear interpolation. A standard approach that the World
Bank itself uses but we are transparent about it and re-run every analysis using
only the real survey points to make sure our conclusions do not depend on the
filled-in values.

---

## Repository Structure

```
ng-growth-poverty/
  data/
    raw/                  # gitignored - API pull cached as parquet
    processed/            # gitignored - processed panel parquet
    README.md             # data source notes
  notebooks/
    eda.ipynb             # data exploration and visualisation
    modelling.ipynb       # statistical analysis and model results
  src/
    config.py             # shared constants and file paths
    loader.py             # World Bank API pull and data validation
    features.py           # data cleaning, interpolation, feature engineering
    models.py             # regression wrappers
  outputs/
    figures/              # all charts
    reports/              # regression results table (CSV)
  pipeline.py             # command line runner for data prep
  environment.yml
  .gitignore
```

---

## Setup

**1. Clone the repo**
```bash
git clone git@github.com:kozah04/ng-growth-poverty.git
cd ng-growth-poverty
```

**2. Create the environment**
```bash
conda env create -f environment.yml
conda activate ng-growth-poverty
python -m ipykernel install --user --name ng-growth-poverty --display-name "ng-growth-poverty"
```

**3. Run the notebooks in order**

The EDA notebook hits the World Bank API on the first run and saves the data
locally. After that, both notebooks run from the saved cache with no internet
needed.

1. `notebooks/eda.ipynb`
2. `notebooks/modelling.ipynb`

---

## Analytical Decisions

These are the choices that shaped the analysis and why we made them.

| Decision | Why |
|---|---|
| Fill gaps in poverty data with linear interpolation | Household surveys happen every 2-7 years. Without filling the gaps we cannot plot trends or run time series regressions. Linear interpolation within the observed range is the standard approach and we validate every result against raw survey points only. |
| Use GDP per capita levels, not growth rates, as the main variable | GDP levels and poverty both trend over time and move together in the long run. Statistical tests confirmed they share a stable long-run relationship (cointegration), which means regressing one on the other is valid. Using growth rates would throw away that long-run information. |
| Take the log of GDP per capita | GDP per capita varies from $200 (Ethiopia) to $6,000 (South Africa) across the panel. Log scale compresses that range and makes the relationship with poverty closer to linear, which is what regression assumes. |
| Use country fixed effects in the panel model | Every country has structural differences - institutions, geography, history - that do not change over time but affect both growth and poverty. Fixed effects absorb all of that, so the model only uses variation within each country over time rather than comparisons between countries. |
| Exclude trade openness from the models | Nigeria has no trade data at all for the full 34-year period. Including it would silently drop Nigeria from any model that uses it, making cross-country comparisons misleading. |
| Re-run every poverty regression on raw survey points | Interpolation is a reasonable assumption but it is still an assumption. If the conclusions change when we use only the 5-8 real data points per country, the findings are fragile. They did not change, which strengthens confidence in the results. |

---

## Limitations

**Sparse poverty data.** We only have 5-8 actual survey observations per country.
The interpolated points between surveys are estimates, not measurements. We handle
this transparently but it caps the precision of the analysis.

**Small panel.** Five countries is enough to run a panel model but not enough to
make strong generalisations. Results should be read as evidence about these five
specific countries, not all of Africa.

**Kenya does not fit the pattern.** Kenya is the only country where poverty rose
as GDP grew and inequality fell simultaneously. This could be a data issue (survey
methodology changed), a composition effect (urban growth masking rural decline) or
a genuine failure of growth to reach the poor. The panel model cannot tell us which.
It is an open question worth investigating separately.

**Correlation, not causation.** The statistical methods we use can identify a
reliable relationship between growth and poverty and can test whether it is stable
over time. They cannot fully rule out other explanations. The findings are robust
associations, not proof that growth causes poverty reduction.

---

## Environment

- Python 3.11
- statsmodels, linearmodels
- pandas, numpy, scipy
- matplotlib, seaborn
- wbgapi, pyarrow
- jupyter, ipykernel

See `environment.yml` for the full dependency list.

---

## References

- World Bank (2024). World Development Indicators.
  https://databank.worldbank.org/source/world-development-indicators
- World Bank (2024). Poverty and Inequality Platform.
  https://pip.worldbank.org
- Engle, R.F. and Granger, C.W.J. (1987). Co-integration and Error Correction:
  Representation, Estimation, and Testing. Econometrica, 55(2), 251-276.
- Granger, C.W.J. (1969). Investigating Causal Relations by Econometric Models
  and Cross-spectral Methods. Econometrica, 37(3), 424-438.
- Brown, R.L., Durbin, J., and Evans, J.M. (1975). Techniques for Testing the
  Constancy of Regression Relationships over Time. Journal of the Royal
  Statistical Society, Series B, 37(2), 149-192.