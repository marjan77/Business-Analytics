# Business-Analytics
To build a model to analyse the price of listed Airbnb’s and determine if the price is reasonable
or overpriced based on other exogenous features from the chosen dataset.
Hosting sites like Airbnb has grown tremendously over the past few years. Airbnb, Inc is a
California based online platform which lists rental sites. It is a household property that holds
rent on a short-term basis to visitors. It has provided temporary housing solutions for a variety
of people over variety of places. The whole platform can be accessed via website and App
which makes it convenient and hassle-free for both hosts and guests. The platform builds profit
by the commission received from the hosts of rental properties.
The presupposition of the group aim is to observe if any of the factors in the data set has
determinative relationship with the price of rentals. Also, the other relationships between
different features in the dataset, if any. The Model will also try to find any non-linear
relationship between the aspects of database and the price. The model should also be able to
classify a new set of data points into 2 categories when a new set of inputs are given.
In the project, the group is trying to build a model that gives suggestions to the host of a
property if the price they gave is reasonable or overpriced. This feedback mechanism will help
the host to price the property competitively so that probability of occupancy is increased. This
is beneficial to both the host and the company. This is because more and more people will be
encouraged to list their property in Airbnb as it provides general trend of market price of similar
properties. This in turn increases the revenue of the business.
# Result
From initial evaluation, the accuracy for random forest and DT boosted were higher and
comparable whereas the other models had slightly lower accuracy.
For precision, the metrics for Random Forest and Decision trees were in the same range. Since
the data set is unbalanced and a false classification cost more than a true classification,
precision and accuracy are not sufficient to compare the models.
Thus, Matthew’s correlation coefficient (MCC) is used and more reliable since it only produces
high score if all the four confusion matrix categories show good results. Random forest and
boosted decision tree have the highest MCC (~0.64) which is approximately 7% higher than
the worst model. A similar result is observed while comparing the area under the Receiver
Operator characteristics curve.
It is also noteworthy that only for logistic and random forest model, the optimum threshold was
above 0.5.
![Screenshot 2022-07-12 185950](https://user-images.githubusercontent.com/96749935/178561451-2ae2315c-b33c-4217-b82b-fb27d856c4e8.png)
