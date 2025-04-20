# EcoSense: Smart Energy and Carbon Tracker - Amrita Vishwa Vidyapeetham
EcoSense is a smart energy meter that integrates IoT, AI and sustainability to  to promote energy efficiency and environmental consciousness in households. 
It is an IoT-based smart monitoring system that detects anomalies in household power consumption and optimizes energy usage using deep learning models deployed on NodeMCU ESP8266.
It is utilizes LSTM, BiLSTM and GRU deep learning model architecture.
## Table of Contents 
- [Abstract](#abstract)
- [Hardware and Software Used](#hardware-and-software-used)
- [Architecture](#architecture)
- [Program Flowchart](#program-flowchart)
- [Block Diagram](#block-diagram)
- [Implementation](#implementation)
- [Evaluation and Results](#evaluation-and-results)
- [Conclusion](#conclusion)
- [Acknowledgement](acknowledgement)
## Abstract
EcoSense is a smart energy and carbon tracking system designed to enhance household energy efficiency and environmental responsibility. It measures real-time power consumption across multiple devices, detects anomalies in power consumption using deep learning networks. It is built around a NodeMCU microcontroller, integrating voltage, current sensors to provide accurate consumption data. Many lightweight deep learning models pre trained with a similar dataset are explored to detect anomalies in power detection. The integrates IoT by using ThingSpeak by Mathworks to display real-time data as user interface. The project uniquely incorporates Tamil Nadu’s specific emission factors to dynamically calculate the carbon impact left behind each and every household.
This project presents Long Short-Term Memory (LSTM), Gated Recurred Unit (GRU) and Reservoir Computing (RC) models. To train all these models, real-time data has been used which includes parameters like voltage, current, power factor and frequency. All the models are trained using 3-fold cross-validation and evaluated based on their performance via accuracy, Mean Squared Error (MSE), R-squared and runtime. 
Through the output we can infer that all the models achieve accuracy above 99% for both training and test data, in which LSTM has highest accuracy as it has the ability to capture long-term dependencies. However, RC stands out for its less runtime and reduced computational power. 
## Hardware and Software Used 
- **Hardware:** ZMP101B Voltage Sensor, ZMPCT103C AC Current Sensor, MCP3008 10-bit A/D converters, NodeMCU ESP8266 Microcontroller, LCD Module (16x2)
- **Software:** MATLAB for developing pre-trained deep learning model, ThingSpeak - deploying the power consumptoion data with responsive HTML page 
## Architecture
Three deep learning models were explored and trained with the same dataset from Kaggle. The accuracy, learning curve etc., were compared to choose a best model for deploying the real-time dataset. 
**Archiecture of each include:**
- **Long Short-Term Memory (LSTM)**
1.	Input Layer: Accepts the feature sequences and each step comprises a 4-dimensional vector (hour, minute, power and frequency).
2.	LSTM Layer: It contains 50 units and assigned ‘OutputMode’ set to ‘last’ to output the final hidden state, which summarizes the entire sequence for classification. 
3.	Fully Connected Layers: Two layers with 32 and 16 units, respectively, each followed by ReLU activation function (f(x)=max (0, x)) to introduce non-linearity to the model.
4.	Dropout Layers: After each fully connected layer 20% dropout rate is applied to prevent overfitting as it randomly deactivates neurons during training.
5.	Output Layer: It is a softmax layer which outputs to class probabilities.
- **Gated Recurrent Unit (GRU)**
1.	Input Layer: Processes the normalized feature sequences (hour, minute, power, frequency).
2.	GRU Layer: It contains 50 units, with ‘OutputMode’ set to ‘last’ to output the final hidden state in a single vector.
3.	Fully Connected Layers: Two layers with 32 and 16 units each and passed into ReLU activation layer to transform the output into a higher-level representation.
4.	Dropout Layers: After each fully connected layers, 20% of dropout rate is applied to prevent the model from overfitting.
5.	Output Layer: It is a fully connected layer with softmax activation for binary classification.
- **Reservoir Computing (RC)**
1.	Input Layer: It accepts the normalized features as a flattened vector per time step, bypassing the need for sequence processing due to the model’s non-recurrent design. 
2.	Fully Connected Layer: It uses a 64-unit layer with ReLU activation layer for RC’s non-linear mapping without requiring a recurrent structure. 
3.	Output Layer: A fully connected layer with softmax as activation function for the respective classification.

## Program Flowchart 
![image](https://github.com/user-attachments/assets/05e194a2-a066-4cd9-a454-8c1d6c184b1a)
(figure 1. Program Flowchart)
## Block Diagram 
![image](https://github.com/user-attachments/assets/01a8a33b-b5b8-4bba-9963-2b041ae37632)
(Figure 2. Block Diagram)
## Implementation  
The microcontroller NodeMCU ESP8266 was connected with voltage and current sensors using DuPont wires. Arduino IDE was used to compute power consumption, carbon footprint and power factor. It was used to connect the same with ThingSpeak. The dataset was obtained from Kaggle which comprises of average current, average voltage, power factor, frequency and timestamps collected from a smart energy meter. 
The following preprocessing was done with the dataset: 
-	Timestamp conversion: converted the timestamps into data time format. 
-	Feature computation: Calculating power as the product of voltage, current and power factor. 
-	Anomaly labelling: By using a statistical threshold, power is classified as “Anomaly” and “Normal”. 
Further, feature extraction and Normalization was done. Extracted hour and minute as temporal features and power-based features. Normalized all the features to range [0,1] using min-max scaling to ensure consistent input scales across models, improving stability. Cross validation for the model used 3-fold cross validation to partition the dataset into training and testing sets, and to provide robust performance. For each fold, approximately two-thirds of the data are used for training and one-third for testing.
Input preperation for for LSTM and GRU, features are organized into sequences, where each sequence represents a series of time steps with 4-dimensional feature vector per step. For RC, features are flattened into a single vector.
The training configration includes using similar configurations for all three models for a fair comparison. The model has been trained using Adam Optimizer with an initial learning rate of 0.0001, 50 epochs and mini-batch size of 64. To prevent over-fitting, dropout regularization (20%) has been applied. Cross-entropy loss function was used to optimize the model.

## Evaluation and Results 
The deep learning model performance is evaluated using similar training configuration for all three models to test their effectiveness in detecting anomalies in smart energy meter data. 
-	Accuracy: In our study, we computed training and testing accuracy to provide baseline measure of overall classification performance. 
-	F1-score: It is the harmonic mean of precision.
-	-	ROC-AUC: It is the area under the receiver operating characteristic curve. It measures the model’s ability to classify. 
-	PR-AUC: It assesses performance by plotting precision against recall across thresholds, focusing on the positive class. 

This section includes a comparison of deep learning networks based on various metrics.

| Metrics         | LSTM  | GRU  | RC   |
|-----------------|-------|------|------|
| Avg. Test Accuracy % | 99.75 | 99.59% | 99.73% |
| Avg. F1-score   | 0.9805 | 0.0985 | 0.9813 |
| Avg. ROC-AUC    | 1.0000 | 1.0000 | 0.9999 |
| Avg. PR-AUC     | 0.7117 | 0.9703 | 0.9932 |
| Avg. Runtime (s) | 28.11 | 46.85 | 9.68 |

(Table 1. Deep Learning networks comparison table)
The results shows that all models achieve high testing accuracies, indicating the model’s robustness in classification task. LSTM outperforms other two models in accuracy, which shows its ability to capture complex temporal dependencies in energy consumption patterns. However, RC outperforms by excelling in having less runtime, approximately three times faster than LSTM, due its simplified architecture which results in efficient prediction. 
![WhatsApp Image 2025-04-20 at 19 11 48_ace83678](https://github.com/user-attachments/assets/e815d49a-398b-432e-bfba-8814603c110d)

(Figure 3. Learning curve of deep learning networks) 
General Observation of the curve: All models (LSTM, GRU, and Reservoir) exhibit a steep decline in training loss during the initial epochs (approximately 0 to 1000). This suggests that the models quickly learn the underlaying patterns in the data during the early stages of training. 
Convergence Behaviour: The models appear to converge after around 2000-3000 epochs, with minimal further reduction beyond this point. This indicates that the models reach point of diminishing returns. The loss stabilizes at a low value (close to 0) for all folds. 
**Deep Learning Models comparison:** 
-	LSTM: The blue lines show consistent convergence across folds, with losses dropping to near 0. The curves are relatively smooth, indicating stable training dynamics. 
-	GRU: The orange lines also converge to a low loss value, with a similar steep initial drop. The GRU curves appear slightly more jagged than LSTM, suggesting minor variability in training stability across folds. 
-	Reservoir: Curves align with LSTM and GRU show similar patterns across three folds, indicating that the models generalize well across different data splits. 

## Conclusion 
The project aimed to design a system to measure and display power consumption of household devices in real-time. To analyse pattern and anomaly detection of devices using deep learning networks. It also aimed to show the power consumption data through an IoT based web application. The project has made a realistic-model of the working principle. 

## Acknowledgement 
We would like to express our heartfelt gratitude to our Dean Dr. Soman K.P. for their invaluable support, guidance, and encouragement throughout the course of this EcoSense: Smart Energy Meter and carbon tracker. 
We are deeply thankful to Dr.Amruta V, professor at school of Artificial Intelligence and Dr.Snigdha Archarya, professor at school of Artificial Intelligence for their constant inspiration and motivation. Their advice and constructive feedback have been essential to the successful completion of this work. 
We would also like to acknowledge Amrita Vishwa Vidyapeetham, school of Artificial Intelligence, for providing the necessary resources and a supportive environment to carry out this work.
Lastly, we are grateful to everyone who contributed, directly or indirectly, to this endeavour.
