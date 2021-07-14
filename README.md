# Palmer Penguins Analysis and ML implementation
A machine learning project with Python applying neural networks from Sci-Kit Learn library to predict three different penguin species.

### Dataset 🚀

The Palmer Penguins’ dataset is a nice alternative to the Iris dataset commonly used in all machine learning examples. This dataset has data for 3 penguin species: Adélie, Chinstrap and Gentoo.


### Built with 📋

* [Sci-kit learn](https://scikit-learn.org/stable/install.html) - Best Machine Learning library for prediction models.

* [Anaconda](https://www.anaconda.com/products/individual) - Recommended to create the venv and then add libraries easily.

* [RStudio](https://www.rstudio.com/products/rstudio/download/) - To export the database and .CSV file from the [original repository](https://github.com/allisonhorst/palmerpenguins).


### Programming language 🖥️

The program was coded with Python 3.0 version in a Jupyter Notebook.

You can download last Python version [here](https://www.python.org/downloads/).



### Algorithm 🔍

The algorithm is divided in 14 stages:
1. Using pandas to read the database from a .CSV file.
2. Write a few functions to know dataframe's nature and column types.
3. Data cleaning by checking which rows had NaN values and dropping them.
4. Shuffling dataframe and resetting indices in order to extract only 70% of data (instead of taking Adélie and Gentoo rows mostly).
5. Saving the whole dataframe in 'backup' and 70% dataframe in 'data'.
6. Creating a LabelEncoder to assign numerical values to the species (Gentoo - 2,   Chinstrap - 1,    Adelie - 0).
7. Setting the multi-layer perceptron by getting train data from 'data' and test data from 'backup'.
8. Applying a StandardScaler for preprocessing train and test data.
9. Training phase with only train data (70%) using a MLPClassifier with 3 hidden layers and 'lbfgs' solver.
10. Predictions with trained data and random values for a penguin.
11. Predictions with trained model for 100% of the dataframe.
12. Declaring a list and populating it with bill length, flipper length and species directly from original dataset.
13. Declaring a list and populating it with bill length, flipper length and species from final_predictions.
14. Calculate precision percentage according to predicted and original data.


### Model precision percentage ⌨️

```
percentage = len(set(final_predictions) & set(species_backup)) / float(len(set(final_predictions) | set(species_backup))) * 100
print("Precision percentage amongst lists: ",percentage)

Precision percentage amongst lists:  100.0
```


## Wiki 📖

You can find more about penguin species in [List of Penguin Species](https://www.birdlife.org/worldwide/news/list-penguin-species)


## Author ✒️

_Francisco Olvera Hernández_

