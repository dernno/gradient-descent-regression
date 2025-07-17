# Linear & Softmax Regression (Gradient Descent)

This project implements **Linear Regression** (univariate and multivariate) and **Softmax Regression** (multiclass classification) using **gradient descent**, inspired by *Lindholm et al., Chapter 5.4*.

---

## Dataset Excerpt 

| x1 (cylinders) | x2 (displacement) | x3 (horsepower) | x4 (weight) | x5 (acceleration) | x6 (year) | x7 (origin) | y (mpg) |
|----------------|-------------------|------------------|--------------|--------------------|------------|--------------|---------|
| 8.0            | 307.0             | 130.0            | 3504.0       | 12.0               | 70         | 1            | 18.0    |
| 8.0            | 350.0             | 165.0            | 3693.0       | 11.5               | 70         | 1            | 15.0    |
| 8.0            | 318.0             | 150.0            | 3436.0       | 11.0               | 70         | 3            | 18.0    |
| 8.0            | 304.0             | 150.0            | 3433.0       | 12.0               | 70         | 2            | 16.0    |
| 8.0            | 302.0             | 140.0            | 3449.0       | 10.5               | 70         | 1            | 17.0    |

This dataset is used for both regression and classification depending on the selected mode.

---

## Usage

Run the main script with the desired mode:

### 🔹 Version 1: Linear Regression (Univariate)

One input feature and one target variable.

```bash
python main.py --mode linear_univariate
```

### 🔹 Version 2: Linear Regression (Multivariate)

Multiple input features and one target variable.

```bash
python main.py --mode linear_multivariate
```
### 🔹 Version 3:  Softmax Regression (Multiclass Classification)

Softmax classifier for 3 classes.

```bash
python main.py --mode softmax
```

## Reference

Lindholm, A., Wahlström, N., Lindsten, F., & Schön, T. B. (2022).  
*Machine Learning: A First Course for Engineers and Scientists*. 1st ed. Cambridge University Press.  
[https://doi.org/10.1017/9781108919371](https://doi.org/10.1017/9781108919371)


