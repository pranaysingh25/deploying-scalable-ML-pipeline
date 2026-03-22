# Model Card: Census Income Prediction

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

**Model Name:** Census Income Binary Classifier
**Model Type:** Random Forest Classifier
**Framework:** Scikit-learn v1.7.2
**Model Parameters:**
- n_estimators: 100
- max_depth: 15
- min_samples_split: 10
- min_samples_leaf: 4
- random_state: 42
- n_jobs: -1 (parallel processing)

**Training Date:** March 2026
**Developer:** Pranay Singh

## Intended Use

This model predicts whether an individual's annual income exceeds $50K based on their demographic and employment information. 

**Primary Use Cases:**
- Income level prediction for research and analysis
- Understanding income determinants based on census data
- Demographics-based income estimation

**Intended Users:** 
- Data scientists and researchers analyzing census data
- Educational purposes in machine learning courses
- Income prediction research

## Training Data

**Data Source:** 1994 Census Income Database
**Dataset File:** `starter/data/census_cleaned.csv`
**Data Cleaning:** Applied whitespace removal, missing value imputation (using mode), and data type conversion

**Data Characteristics:**
- Total Records: 32,561
- Training Set: 26,048 (80%)
- Test Set: 6,513 (20%)
- Train-Test Split: Random split with random_state=42

**Features:**
- **Demographic:** age, race, sex, native-country
- **Socioeconomic:** education, marital-status, relationship, occupation, workclass
- **Work-Related:** hours-per-week, employment-focused numeric features
- **Capital:** capital-gain, capital-loss, final-weight (fnlgt)

**Target Variable:** salary (binary: <=50K or >50K)

**Data Preprocessing:**
1. Removed leading/trailing whitespace from all columns and values
2. Imputed missing values (represented as '?') with mode:
   - workclass: imputed with 'Private' (5.64% missing)
   - occupation: imputed with 'Prof-specialty' (5.66% missing)
   - native-country: imputed with 'United-States' (1.79% missing)
3. Converted numeric columns to appropriate data types
4. Applied One-Hot Encoding to categorical features
5. Applied Label Binarization to target variable

## Evaluation Data

**Test Set:** 6,513 samples (20% of total data)
- Same preprocessing applied as training data
- Held out from model training
- Used for evaluation metrics

**Data Split Method:** 
- Stratified random train-test split would be recommended for future iterations
- Current split: random split with 80-20 ratio

## Metrics

**Classification Metrics Used:**
- Precision: Proportion of positive predictions that were correct
- Recall: Proportion of actual positives correctly identified
- F-Beta Score (β=1): Harmonic mean of precision and recall

**Model Performance:**

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| Precision | 0.8129 | 0.7840 |
| Recall | 0.5871 | 0.5684 |
| F-Beta (β=1) | 0.6818 | 0.6590 |

**Performance Observations:**
- The model shows strong precision (78.40% of predicted high earners actually earn >50K)
- Recall of 56.84% indicates the model identifies approximately 57% of actual high earners
- Minimal gap between training and test metrics suggests good generalization without significant overfitting
- Model trades off precision for moderate recall, prioritizing prediction accuracy over coverage

## Ethical Considerations

**Potential Biases:**
- The dataset is from 1994 and may have outdated demographic representations
- Historical socioeconomic disparities are reflected in the data
- Income predictions may disproportionately advantage certain demographic groups

**Fairness Concerns:**
- Model performance varies across demographic slices (race, sex, country of origin)
- Occupation-based features may perpetuate historical employment discrimination
- Education-based features may reflect historical educational access disparities

**Recommendations for Fairness:**
1. Conduct detailed fairness analysis across demographic slices
2. Monitor disparate impact across protected attributes
3. Consider fairness constraints when deploying the model
4. Regular audits of model predictions across demographic groups
5. Transparent communication about model limitations

**Privacy Considerations:**
- Model trained on census data containing personal information
- Ensure GDPR/CCPA compliance in model deployment
- Implement access controls for model endpoints
- Consider differential privacy techniques for future iterations

## Caveats and Recommendations

**Limitations:**
1. **Temporal Limitations:** Model trained on 1994 data; income dynamics have changed significantly
2. **Feature Limitations:** Does not include important modern income determinants (tech skills, certifications, etc.)
3. **Class Imbalance:** Income >50K is less frequent in dataset (likely imbalanced)
4. **Geographic Bias:** Limited to US data; not generalizable internationally
5. **Model Complexity:** Random Forest lacks interpretability; difficult to explain predictions

**Recommendations for Improvement:**
1. **Update Training Data:** Retrain on more recent census data (2020+)
2. **Add Features:** Include modern income determinants (online certifications, tech skills, remote work, etc.)
3. **Class Weight Handling:** Use class weights or sampling techniques for imbalanced data
4. **Fairness Improvements:** 
   - Conduct thorough demographic parity analysis
   - Implement fairness constraints during training
   - Use fairness-aware algorithms if disparities exist
5. **Model Interpretability:**
   - Generate SHAP values for feature importance
   - Create partial dependence plots
   - Document feature engineering decisions
6. **Validation Enhancements:**
   - Use cross-validation instead of simple train-test split
   - Perform fairness evaluation across demographic groups
   - Test on held-out temporal data
7. **Deployment:**
   - Implement model monitoring for performance drift
   - Set up fairness monitoring pipelines
   - Create feedback loops for continuous improvement

**Out-of-Scope Use Cases:**
- Real-time decision making for critical applications (hiring, lending)
- Use with individuals outside the US census distribution
- Prediction on data significantly different from 1994 census characteristics
- Regulatory or legal decision-making without human review

**Implementation Notes:**
- Model requires preprocessing with the trained OneHotEncoder and LabelBinarizer
- Categorical features must be encoded using the exact encoder from training
- Continuous features are not scaled; can be enhanced with standardization
- Model can be loaded from `model/model.pkl` for inference

**Contact:** For questions about this model, contact me directly.
