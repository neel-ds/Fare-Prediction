# Fare Prediction
The transportation industry is thriving in the hotspot city. Such 
companies 
are providing ridesharing services from door to door using algorithms or 
chunks of data to leverage the customers at a valuable price. Generally, 
those fares are dynamic and estimated considering the distance but the 
target fare should not be dependent only on trip distance. It must 
consider 
significant features like the demands, traffic, and potential area based 
on 
data from past years. So, the approach considers the service provider and 
customer to get a relative benefit over the competition. Likewise, the 
proposed method will include end-to-end development for real-world usage.

**Evaluation and Conclusion**
| Model Name | RMSE | R2 Score |
| :-----: | :-----: | :-----: |
| Linear Regression | 6.64 | 54.94% |
| Decision Tree | 5.20 | 71.86% |
| Random Forest | 5.26 | 71.49% |
| Gradient Boosting | 4.80 | 76.22% |

The purpose of this project is to estimate those fare calculations, with a 
machine learning approach that could be fast enough to utilize while 
providing services to the real-time customer. The results of the 
regression 
showed that the proposed method has potential as far as different known 
quality metrics. Furthermore, I implemented feature scaling and the 
performance improved. In the deployment, the Gradient Boosting model  
performed well with 4.47 RMSE & R2 score of 79.22%. The model has been 
registered in the MLflow so it can be served in real-time as a REST 
endpoint which allows HTTPS requests and gives the output based on the 
input request by the client. Later models can be more robust and precise 
by utilizing real-time traffic data. To sum up, test evaluation and 
outright insights can be implemented in the food delivery chain or other 
product delivery services strategically as per the requirement and domain 
expertise.

Kindly read the project [documentation 
file](https://github.com/neel-ds/Fare-Prediction) for getting started. For 
more details of the project, refer the detailed [report 
file](https://github.com/neel-ds/Fare-Prediction). To download the dataset 
[here](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction/data).

Copyrights © 2022 by Neel Patel.
All rights are reserved.
