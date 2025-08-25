## Model Selection and Justification

After analyzing the six trained models, the chosen model for production is:

**XGBoost with the Top 10 features and class balancing (`scale_pos_weight`).**

### Arguments

1. **Business Objective**  
   The key requirement is to **detect flight delays**.  
   Models without balancing achieve high accuracy (~81%) but fail to identify delays (recall close to 0).  
   This makes them practically useless for the problem at hand.  

2. **Handling Class Imbalance**  
   Delays represent only ~19% of the dataset.  
   By applying class balancing, the model pays more attention to the minority class and significantly improves its performance in detecting delays.  

3. **Performance of the Chosen Model**  
   - Recall for class "delay" (1): **~0.69**, meaning it detects 69% of actual delays.  
   - Accuracy drops to ~55%, but this trade-off is acceptable because detecting delays is more important than minimizing false alarms.  

4. **Why XGBoost Over Logistic Regression**  
   While both algorithms performed similarly, **XGBoost** is more robust for tabular data with non-linear relationships.  
   Additionally, it provides feature importance, which improves interpretability for stakeholders.  

5. **Top 10 Features**  
   Reducing the dataset to the 10 most important features simplifies the model without losing performance.  
   This improves efficiency and interpretability.  

---

### ✅ Final Conclusion
The **best model for production** is:  
**XGBoost trained with the Top 10 features and with class balancing.**

This model strikes the right balance by **maximizing the detection of delays (high recall for the minority class)** while maintaining reasonable overall performance.