


# California-Housing-ML-Models

This project aims to predict house prices in California based on various features such as median household income, house age, and location using machine learning techniques. The goal is to build an accurate regression model that can assist in estimating property values.


## Implementation Details

- Dataset: California Housing Dataset (view below for more details)
- Model: [Linear Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- Input: 8 features - Median Houshold income, House Area, ...
- Output: House Price

## Dataset Details

This dataset was obtained from the StatLib repository ([Link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html))

This dataset was derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average number of rooms and bedrooms in this dataset are provided per household, these columns may take surprisingly large values for block groups with few households and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing function.

- [California Housing Dataset in Sklearn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
- 20640 samples
- 8 Input Features: 
    - MedInc median income in block group
    - HouseAge median house age in block group
    - AveRooms average number of rooms per household
    - AveBedrms average number of bedrooms per household
    - Population block group population
    - AveOccup average number of household members
    - Latitude block group latitude
    - Longitude block group longitude
- Target: Median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

## Evaluation and Results
![Regression Results](results/regression_results.png)

The regression model achieved the following performance metrics:

- R2 Score: 0.45 -049
  

These results indicate that the model provides a reasonably good fit to the data, with an R2 score of 0.78 indicating that 78% of the variance in house prices is explained by the model.

## Key Takeaways

Throughout this project, we gained insights into feature importance, handled missing data, and fine-tuned the model to improve its performance. Challenges included dealing with outliers and selecting appropriate evaluation metrics.


## How to Run

1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Open and run the Jupyter notebook `California_Housing_Prediction.ipynb`.

## Roadmap

Future enhancements for this project may include:
- Experimenting with different regression models.
- Implementing feature selection techniques.
- Optimizing hyperparameters.

## Libraries 

**Language:** Python

**Packages:** Sklearn, Matplotlib, Pandas, Seaborn


## FAQ

#### How does the linear regression model work?
Linear regression models the relationship between the input features and the target variable using a linear equation.

#### How do you train the model on a new dataset?
To train the model on a new dataset, follow the same steps outlined in the Jupyter notebook with the new data.

#### What is the California Housing Dataset?
The California Housing Dataset contains information about housing prices in California based on data from the 1990 U.S. census.

## Acknowledgements

All the links, blogs, videos, papers you referred to/took inspiration from for building this project. 

- [Scikit-learn](https://scikit-learn.org/)
- [California Housing Dataset Source](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html)

## Contact

If you have any feedback/are interested in collaborating, please reach out to me at tejashkumarsn68.com


## License

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

