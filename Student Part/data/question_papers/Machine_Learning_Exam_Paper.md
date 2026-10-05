# SVKM's NMIMS – Mukesh Patel School of Technology Management & Engineering
## Examination Paper: Machine Learning
**Academic Year:** 2024-2025 (Final Examination / Re-Examination 2023-24)
**Program:** B Tech / MBA Tech / BTI Computer
**Stream:** CE, COMP, CS, EXTC
**Semester:** VI / X (Year III/V)
**Date:** 07/05/2025
**Duration:** 3 hours (2:00 pm to 5:00 pm)
**Total Marks:** 100
**Number of Pages:** 4

---

## Examination Instructions
1. Question No. 1 is compulsory.
2. Out of remaining questions, attempt any 4 questions.
3. In all 5 questions to be attempted.
4. All questions carry equal marks.
5. Answer to each new question to be started on a fresh page.
6. Figures in brackets on the right hand side indicate full marks.
7. Assume suitable data if necessary.

---

## Question 1 – Answer Briefly [Total: 20 Marks]

**Q1.A** [CO-1; SO-1; BL-2] [5 Marks]
Discuss the five important factors to consider when selecting a machine learning model for a given problem statement.

**Q1.B** [CO-1; SO-1; BL-4] [5 Marks]
Explain the different techniques for handling null values in a dataset. Apply any three of the techniques in the dataset given below. Explicitly mention which method is used for the particular column.

| ID | Name | Age | Salary | Department | Experience |
|----|------|-----|--------|------------|------------|
| 1 | Alice | 25 | 50000 | HR | 2 |
| 2 | Charle | 30 | NaN | IT | NaN |
| 3 | Bob | NaN | 60000 | HR | 5 |
| 4 | David | 40 | NaN | NaN | 10 |

**Q1.C** [CO-2; SO-1; BL-2] [5 Marks]
Explain how the classification problem is solved by using the Logistic Regression algorithm.

**Q1.D** [CO-2; SO-2; BL-2] [5 Marks]
What is a recommender system? Give one example.

---

## Question 2 [Total: 20 Marks]

**Q2.A** [CO-2; SO-2; BL-4] [10 Marks]
Consider a set of data:

| x | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| y | 2 | 3 | 5 | 4 | 6 |

Find the equation for linear regression model (y = mx + b) using gradient descent after two iterations. Consider initial values for m = 1, b = 2, and learning rate = 0.001.

**Q2.B** [CO-2; SO-2; BL-2] [10 Marks]
Describe the role of hidden layers in MLP and how the backpropagation algorithm is used for training the network. Provide a basic explanation of the weight update process in backpropagation.

---

## Question 3 [Total: 20 Marks]

**Q3.A** [CO-3; SO-6; BL-3] [10 Marks]
A medical diagnostic team aims to automate the decision-making process for diagnosing a disease based on patient symptoms. Use the following dataset to develop a classification tree that predicts whether a patient is likely to have the disease or not. The model will utilize Entropy and Information Gain in a Decision Tree to identify the most influential symptoms and improve the diagnostic process.

| Fever | Cough | Disease (Yes/No) |
|-------|-------|-----------------|
| High | Mild | No |
| High | Severe | No |
| Normal | Mild | Yes |
| Low | Mild | Yes |
| Low | Severe | No |
| Normal | Severe | Yes |
| High | Mild | Yes |

**Q3.B** [CO-1; SO-1; BL-2] [10 Marks]
Write Advantages of Hierarchical Clustering over K-means Clustering.

---

## Question 4 [Total: 20 Marks]

**Q4.A** [CO-2; SO-2; BL-2] [10 Marks]
Explain Multiple regression and how it differs from linear regression. When should Multiple regression be used, and what are its advantages and limitations?

**Q4.B** [CO-3; SO-6; BL-6] [10 Marks]
Build a Naïve Bayes classifier for the given dataset. Also, test the classifier for a given new instance (Age=Youth, Income=Medium, Student=Yes, Credit rating=Fair).

| RID | Age | Income | Student | Credit Rating | Class: buys computer |
|-----|-----|--------|---------|---------------|----------------------|
| 1 | Youth | High | No | Fair | No |
| 2 | Youth | High | No | Excellent | No |
| 3 | Middle-aged | High | No | Fair | Yes |
| 4 | Senior | Medium | No | Fair | Yes |
| 5 | Senior | Low | Yes | Fair | Yes |
| 6 | Senior | Low | Yes | Excellent | No |
| 7 | Middle-aged | Low | Yes | Excellent | Yes |
| 8 | Youth | Medium | No | Fair | No |
| 9 | Youth | Low | Yes | Fair | Yes |
| 10 | Senior | Medium | Yes | Fair | Yes |
| 11 | Youth | Medium | Yes | Excellent | Yes |
| 12 | Middle-aged | Medium | No | Excellent | Yes |
| 13 | Middle-aged | High | Yes | Fair | Yes |
| 14 | Senior | Medium | No | Excellent | No |

---

## Question 5 [Total: 20 Marks]

**Q5.A** [CO-2; SO-2; BL-2] [10 Marks]
Discuss the computational challenges of Ensemble Learning algorithm. Justify your answer with an example.

**Q5.B** [CO-4; SO-6; BL-3] [10 Marks]
Consider following instance given as input to K-Means algorithm for k = 3. Find members of these 3 clusters after one iteration using Euclidean distance metric.
X = {(2,10), (2,5), (8,4), (5,8), (7,5), (6,4), (1,2), (4,9)}.
Assume initial cluster centroids are c1(2,10), c2(5,8), c3(1,2).

---

## Question 6 [Total: 20 Marks]

**Q6.A** [CO-2; SO-2; BL-5] [10 Marks]
A machine learning engineer is building a regression model to predict the stock prices of five companies. After testing the model, the engineer compared the predicted stock prices with the actual stock prices (in ₹) as follows:

| Company | Actual Price (₹) | Predicted Price (₹) |
|---------|-----------------|---------------------|
| A | 150 | 140 |
| B | 220 | 200 |
| C | 175 | 180 |
| D | 300 | 260 |
| E | 190 | 210 |

Using the data above, define and calculate the Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for the model. Then, explain how these metrics differ in terms of sensitivity to large errors.

**Q6.B** [CO-4; SO-6; BL-4] [10 Marks]
A manufacturing company has implemented a machine learning model to automatically detect defective products on the assembly line. After running the model on a batch of items, the following results were observed: Out of all the products that were actually defective, the model correctly identified 6 as defective, while 3 defective items were wrongly classified as non-defective and passed through quality check. Among the non-defective products, 8 were correctly labeled as good, while 3 were wrongly flagged as defective, even though they were fine.
Based on these details:
1. Create confusion matrix
2. Define and compute (i) Accuracy (ii) Precision (iii) Recall (iv) F1 Score

---

## Question 7 [Total: 20 Marks]

**Q7.A** [CO-2; SO-2; BL-2] [10 Marks]
A healthcare company is developing an AI-based diagnostic system to predict the likelihood of a patient having a specific disease based on medical test results. Initial experiments using a single Decision Tree classifier resulted in high variance, leading to inconsistent predictions on new data. To improve predictive performance and reduce overfitting, the company decides to implement an ensemble learning approach using multiple models. Explain how Bagging (Bootstrap Aggregating) can be applied to enhance the model's accuracy and stability.

**Q7.B** [CO-1; SO-2; BL-2] [5 Marks]
How the bias-variance trade-off influences the performance of a machine learning algorithm.

**Q7.C** [CO-1; SO-6; BL-2] [5 Marks]
Describe One-Hot Encoding and Ordinal Encoding with Examples.

---

## Course Outcome (CO) Reference
- CO-1: Identify machine learning techniques suitable for a given problem
- CO-2: Solve the problems using various machine learning techniques
- CO-3: Develop an application using machine learning techniques
- CO-4: Evaluate and interpret the results of the algorithms

## Bloom's Level (BL) Reference
- BL-1: Remember | BL-2: Understand | BL-3: Apply | BL-4: Analyze | BL-5: Evaluate | BL-6: Create
