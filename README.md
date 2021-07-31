# Traffic-Incident-Statistical-Analysis-Python
#### This was a proprietary project therefore the input and output data are unavailable at this time.

## Background and Goal
From 2016-2020, company employees were involved in 400+ auto incidents every year globally. The purpose of the data analysis in this project is to better protect employees 
and their families by using our current data to build a prescriptive model and prevent/reduce the number of incidents employees are involved in.

## Problem Statement
Is it possible to identify the causation of these incidents? 
If so, we can create a predictive model to identify the probability of a driver getting into an incident by assigning a “risk score”.
From this, we may be able to prevent an incident from occurring by prescribing specific actions (e.g., take specific driver trainings) to each driver based upon their risk score.

## Hypothesis
The relationship between each employee’s driver training taken/not taken and the total associated number of incidents can explain the causation of incidents because this 
category is a company controlled attribute of the driver, i.e., the company can assign/remove/change additional trainings. 

## The Data
Two datasets (training_file.csv; accident_file.csv) were used in this analysis, to uncover key correlations in the data regarding the number of incidents. The data includes 
every employee driver who was or was not involved in a traffic incident from 2016-2020. Many attributes were analyzed, such as location, road conditions, weather conditions, 
and tenure. 

## Analysis Technique
Before moving directly to the prescriptive model phase, it was essential to perform regression analysis on the data to determine its reliability in explaining the number of 
incidents occurred. Basically, regression analysis determines whether the current data we have is enough to move forward with a prescriptive machine learning model or if
there are additional external attributes that must be captured. 

## Results
The hypothesis was rejected – driver trainings taken/not taken do not explain the causation of auto incidents. An R-squared value of 0.290 was identified which is very 
small (R-squared value indicates strong or weak predictive power). 0.290 means that only 29% of the total number of incidents can be explained from the training data - 
this is not high enough to rely upon in a prescriptive model as it will provide grossly inaccurate predictions. 
Simply put, external factors that are not captured in the current data play a more crucial role in explaining the causes of the incidents. 

## Recommended Next Steps
Currently, we are only capturing data points when a driver is involved in a traffic incident. 
In other words, we are not capturing driver data on “normal days”, or when drivers do NOT get into a traffic incident. 
To understand the historical patterns of the factors that contribute or do not contribute to an incident in order to build a reliable predictive model, we need to capture 
data points around (but not limited to) the following factors on days a driver did NOT get into a traffic incident:

1. Road and weather conditions on days each driver drove but did NOT get into an accident. 
2.	Specific routes on days a driver drove and did get into an accident.
3.	Specific routes on days a driver drove and did NOT get into an accident.
4.	Public auto incident rates on company driver routes and locations.
5.	Traffic conditions on company driver routes and locations.
