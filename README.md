# Titanic Survivor Prediction

This project uses the Titanic passenger dataset to predict whether a passenger survived. The notebook prepares the data, creates additional features, converts the categorical values into numerical values, and trains a Random Forest classifier.

The main idea was to start with the original passenger information and gradually transform it into features that can be used by a machine learning model. Each step is inspected before moving to the next one so that the choices made during preprocessing are based on the data.

## Project files

- `titanic_Predictor.ipynb` - the complete data preparation, feature engineering and model training process
- `train.csv` - training data, including the `Survived` target column
- `test.csv` - test data without the target column

## How to run the project

1. Install Python with Jupyter Notebook or JupyterLab.
2. Install the required packages:

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

3. Keep `train.csv`, `test.csv` and `titanic_Predictor.ipynb` in the same folder.
4. Open and run the notebook from top to bottom.

## Notebook workflow

### 1. Loading the data

The notebook first loads the train and test datasets and displays them.

The train data contains `Survived`, which is the value that the model must learn to predict. The test data does not contain this column. Displaying both datasets at the beginning makes it possible to check their columns and structure before changing anything.

### 2. Separating the target

`Survived` is saved in a separate variable and removed from the training features.

This is important because the target must not be used as an input feature. If it stayed in the feature data, the model could use the answer while trying to learn the answer.

The train and test features are then combined temporarily. This allows the same preprocessing steps to be applied to both datasets. The original train and test passenger IDs are kept so the data can be separated again later.

### 3. Checking and filling missing values

The notebook checks how many missing values exist in every column before deciding how to fill them. Different columns describe different types of information, so one replacement method is not suitable for every column.

#### Age

The passengers are grouped by `Pclass` and the median age of each class is calculated.

The median is used because it is less affected by very young or very old passengers than the mean. Using the median age of the corresponding ticket class also uses information about the passenger instead of using one value for everyone.

#### Fare

The missing fare is estimated from passengers with the same `Pclass` and `Embarked` value.

Passengers in the same class and from the same port are more comparable than all passengers in the dataset. The median fare of this group is then used for the missing value.

#### Embarked

The missing embarkation value is investigated using passengers with a similar first-class fare. The most common embarkation port in this group is used to fill the missing categorical value.

The most common value is used because `Embarked` is categorical and cannot be filled with a mean or median.

#### Cabin

Most passengers do not have a cabin value. These missing values are grouped under `M`.

For passengers with a cabin, only the first character is kept. This represents the cabin deck and produces a smaller feature than keeping the complete cabin number. The rare `T` value is also grouped with `M` because it appears only once and would not provide a reliable separate group.

### 4. Feature engineering

The next part creates new features that contain more useful information than some of the original columns.

#### Passenger titles

The `Name` column is split to find the passenger title, such as `Mr`, `Mrs`, `Miss` or `Dr`.

Rare titles are combined into broader groups such as `Military` and `Nobility`. This keeps information about social role or status while avoiding too many categories with very few examples.

#### Age groups

The minimum and maximum ages are checked first. The ages are then divided into intervals using `pd.cut`.

The age groups provide a categorical representation of age. This can make it easier for the model to identify patterns between children, young adults and older passengers.

#### Passengers per ticket

The number of passengers using each ticket is calculated.

Passengers with the same ticket may have travelled together, so this feature can give information about the size of the travelling group.

#### Fare per ticket

The fare is divided by the number of passengers sharing the same ticket.

This creates an approximate individual fare and can be more informative than the total ticket fare when several passengers travelled together. The resulting values are also divided into groups.

#### Family size

`SibSp` and `Parch` are added together and one is added for the passenger:

```text
Number_of_Family = SibSp + Parch + 1
```

The extra one is needed because `SibSp` and `Parch` count the other family members, not the passenger themselves.

### 5. Removing the original columns

The original `Name`, `Age`, `Ticket`, `Fare`, `Ticket_Count` and `Fare_per_Ticket` columns are removed after the new features have been created.

The new features keep the information selected from these columns in a simpler form. Removing the long text and high-cardinality columns also makes the final feature set more suitable for the model.

The number of unique values in the remaining columns is checked before training.

### 6. Preparing values for the model

`PassengerId` is used as the index so that the original train rows can be selected from the combined data.

The categorical values are first converted to strings and encoded as numbers using `LabelEncoder`. One-hot encoding is then applied to `Sex`, `Embarked` and `Title`, creating separate columns for their categories.

This is done because the Random Forest implementation expects numerical input. One-hot encoding also prevents the categories in these columns from being interpreted as an artificial numerical order.

### 7. Scaling and splitting the data

The training features and the `Survived` values are separated into `X` and `y`.

The features are scaled with `MinMaxScaler`, and then the data is split into training and test groups using an 80/20 split. The `random_state` is fixed at `0` so that the same split can be reproduced.

The shapes of the resulting arrays are printed to verify that the split was performed correctly.

### 8. Training the first Random Forest

First, a Random Forest classifier is trained with its default parameters.

This gives a baseline accuracy. A baseline is useful because later parameter searches can be compared with a simple initial model.

### 9. Randomized parameter search

Several possible values are created for parameters such as:

- `criterion`
- `n_estimators`
- `max_depth`
- `min_samples_split`
- `max_features`
- `max_samples`

`RandomizedSearchCV` tests random combinations using five-fold cross-validation. This is a faster way to explore a large parameter space and find a promising region for the model parameters.

The best parameters and the cross-validation training accuracy are printed. The selected model is also evaluated on the held-out test data.

### 10. Grid parameter search

The second search uses the parameters found by the randomized search.

Smaller ranges are created around those values, and `GridSearchCV` checks every combination in this reduced search space. This is a more focused search and can refine the result without checking the entire original parameter space again.

The final parameters and the train and test accuracy are printed at the end.

## Recorded results

The notebook currently contains these recorded results:

| Model stage | Cross-validation accuracy | Test accuracy |
|---|---:|---:|
| Initial Random Forest | - | 0.8268 |
| RandomizedSearchCV | 0.8286 | 0.8268 |
| GridSearchCV | 0.8076 | 0.8212 |

The randomized search result is slightly better than the final grid search result in the recorded run. This shows why both training and held-out test accuracy should be checked instead of assuming that a more detailed search will always produce a better test score.

## Notes

- The notebook is written as an exploratory workflow, so intermediate tables, value counts and shapes are displayed after important steps.
- The train and test data are combined during preprocessing and separated by `PassengerId` for the training stage.
- Running the notebook again can produce different search results if the code or input data changes, so the displayed results should be treated as the results of the recorded run.
