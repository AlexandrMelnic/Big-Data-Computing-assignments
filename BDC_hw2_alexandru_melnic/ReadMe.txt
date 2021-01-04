In the notebook ReviewsClustering all the code is executed and the analysis is performed.
In reviews_cleaner is present the class that performs the import of the texts and the cleaning of 
them. 
utils contains functions needed to compute the metrics, take the most relevant terms
for each clustering and plot the wordclouds.

The import and cleaning of the texts is executed from the TextProcessing class. First it is 
necessary to instantiate the class passing as arguments the paths of the texts and labels and 
optionally also other parameters as commented in the code and in the report. To start the import
and cleaning it is necessary to use the class method import_and_clean. The cleaned texts and the 
respective labels are given by the class attributes processed_texts and true_target.
The vectorization and clustering are performed in the notebook.