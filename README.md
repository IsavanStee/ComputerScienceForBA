# Computer Science For Business Analytics
Product Duplicate Detection Across Web Shops: A Scalable Approach Using Locality Sensitive Hashing

Welcome to the implementation of LSH, which serves as a pre-selection technique in order to find the duplicates by applying classification.
The code of this project is written in collaboration with Nouhaila Zaguaj (572506nz). 

# Data
- The data set can be found in the file: TVs-all-merged.zip
- The data set consists of 1,624 descriptions of televisions coming from four online Web stores: Amazon.com, Newegg.com, Best-Buy.com, and TheNerds.net.
- Each product in the data set is described by a title and has additional data kept in key-value pairs.

# Implementation
The methodology used to find a scalable solution for product duplicate detection is implemented as follows.
The initial step of the process is data cleaning, then followed by extracting the models words and the brand from the title of the product in order to create binary representations for every product. Following that, Locality Sensitive Hashing (LSH) is applied in order to find candidate pairs. Finally, the Single Similarity Method (SSM) is utilized to identify the true duplicates from the candidate pairs. For the SSM technique a different approach is used than the general MSM method according to Van Bezu et al. (2015). In this project, the method utilized to find the duplicates is based on the Jaccard Similarity measure and instead of a clustering based-approach, classification is utilized. 
Bootstrapping is used as an evaluation technique, in order to obtain robust results. For each bootstrap, 63% of the data is used for training (to tune the parameters). The remaining data is considered as the test data used for the evaluation. In total, 6 bootstraps are considered in the evaluation procedure. The final performance of each metric is calculated as the average across all bootstrap iterations.

An overview of the method can be found in the file: Overview method.

In the program, the number of hashes is specified to be approximately 50% of the size of the binary vector representations of the products. Additionally, the threshold for the Jaccard Similarity in the SSM method is set to 0.8. 


[1] van Bezu, R., Borst, S., Rijkse, R., Verhagen, J., Frasincar, F., Vandic, D.: Multi-
component similarity method for web product duplicate detection. In: 30th Sympo-
sium on Applied Computing (SAC 2015). pp. 761–768. ACM (2015)
