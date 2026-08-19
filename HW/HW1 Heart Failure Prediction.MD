## There are three CSV files which will be the input data in this homework.

### events.csv: (pid, event_id, vid, value)
- pid: De-identified patient identier.
- event_id: Clinical event identifier.
- vid: Visit identifier.
- value: Contains the value associated to an event (always 1 in the synthesized dataset).

### hf_events.csv: (pid, vid, label)
- vid: the index of the first visit with heart failure of the patient
- label: 1 indicates the presence of heart failure.

### event_feature_map.csv: (idx, event_id) 
- a map from an event_id to an integer index

## 1 Descriptive Statistics
- Event count: Number of events recorded for a given patient.
- Encounter count: Number of visits recorded for a given patient.

## 2 Feature construction
![](./images/example.png)

- Index vid: Index vid is evaluated as follows:
- For heart failure patients: Index vid is the vid of the first visit with heart failure for that patient (i.e., vid field in hf_events.csv).
- For normal patients: Index vid is the vid of the last visit for that patient (i.e., vid field in events.csv).
- Observation Window: The time interval you will use to identify relevant events. Only events present in this window should be included while constructing feature vectors.
- Prediction Window: A fixed time interval that is to be used to make the prediction.

### 2.1 Compute the index vid

### 2.2 Filter events
#### Remove the events that occur outside the observation window. That is, only keep events in visits before index vid. 

### 2.3 Aggregate events
#### Step 1: Convert string event IDs (e.g., DIAG_244, LAB_50931) to numeric feature_id using event_feature_map.csv
#### Step 2: Calculate event counts per patient.
#### Step 3: Normalize event counts to [0, 1] range (Min-Max scaling).

### 2.4 Save in SVMLight format
#### Construct a feature vector and represent the feature vector in SVMLight format.

'''
<line> .=. <target> <feature>:<value> <feature>:<value>
<target> .=. 1 | 0
<feature> .=. <integer>
<value> .=. <float>
'''

- The target value and each of the feature/value pairs are separated by a space character. 
- Feature/value pairs MUST be ordered by increasing feature number. 
- Features with value zero can be skipped.

## 3 Predictive Modeling
### 3.1 Model Creation
### Step A. Classifiers (Logistic Regression, SVM and Decision Tree) on a training dataset in features_svmlight.train
### Step B. Evaluate your predictive models on a test dataset in features_svmlight.val

### 3.2 Model Validation
#### In order to fully utilize the available data and obtain more reliable results, machine learning practitioners use cross-validation to evaluate and improve their predictive models. 
#### Two cross-validation strategies against SVM.
- K-fold: Use the number of iterations k=5
- Randomized K-fold: Use a test data percentage of 20% and k=5 for the number of iterations for Randomized
