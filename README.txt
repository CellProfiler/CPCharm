== Repository content ==

* Modules folder: CP-CHARM-specific modules that are not yet included in latest CellProfiler release. The use of these modules outside of CP-CHARM pipelines is not recommended before their inclusion in CellProfiler's official release.

* Pipelines folder: Ready-to-use CellProfiler pipelines for extraction of the CHARM-like feature vector.

* Classifier folder: Python scripts for training and testing a classifier, as well as for further label-free classification. 
 
* Example folder: Application example of CP-CHARM. The folder contains:
			- "Input_Images", a folder containing images from a 2-class problem (Negative and Positive)
			- "DefaultOUT_CHARM-like_training_data.csv" and "DefaultOUT_CHARM-like_training_labels.csv", the feature vectors and corresponding labels for all images extracted using the "[TrainTestMode]CHARM-like.cp" pipeline
			- "pcalda_classifier.pk", a pca-lda classifier trained and tested using 10-fold cross validation with the measurements from "DefaultOUT_CHARM-like_training_data.csv"
			- "results_summary.txt", a report of "pcalda_classifier.pk" training and testing phase
			- "wnd_classifier.pk", a WND classifier trained and tested using WND-CHARM's custom validation method with the measurements from "DefaultOUT_CHARM-like_training_data.csv"
			- "predicted_labels.csv", the output of blind classification of "Input_Images" using the previously trained and tested "pcalda_classifier.pk" classifier


== Setting up CP-CHARM-like ==

1) Download compiled version of CellProfiler

2) Install modules specific to CP-CHARM
	* Download the Modules folder with all its content
	* In CellProfiler's Preference menu, set up the plugin directory to point to your downloaded Modules folder
	* Note that you must restart CellProfiler after modifying the plugin directory path


== Running CP-CHARM-like ==

1) Extracting features in CellProfiler
	* Two pipelines are available, located in the Pipelines folder:
		- "[TrainTestMode]CHARM-like.cp" gathers metadata on filename (image ID) and on folder name (class) and outputs two .csv files ("DefaultOUT_CHARM-like_training_data.csv" and "DefaultOUT_CHARM-like_training_labels.csv")
		- "[ClassifyMode]CHARM-like.cp" gathers metadata on filename (image ID) only and outputs one .csv file ("DefaultOUT_CHARM-like_data.csv")

	* For either pipeline, either open CP, load the "CHARM-like.cp" pipeline you want and run analysis, or run CP headless (from CellProfiler_Head directory) using the following command:

	python CellProfiler.py -p ../CHARM-like.cp -c -r -o ../ -i ../Input_Images --plugins-directory=./cellprofiler/modules/plugins/

2) Training and testing using traintest.py script, located in the Classifier folder
	* Severals options are available using the following command line (from the Classifier folder): 

	python traintest.py [DATA_FILE (.csv)] [LABELS_FILE (.csv)] [OUTPUT_PATH] [NB_RUNS] [HOLD OUT SAMPLES (1:yes, 0:no)] [DISPLAY CONFUSION MATRIX (1:yes, 0:no)] [CLASSIFICATION_METHOD ("lda", "wnd")] [VALIDATION_METHOD ("save25", "kfold")] [NB_FOLDS (if VALIDATION_METHOD = "kfold")]

	* To run CP-CHARM-like with the default output from the "[TrainTestMode]CHARM-like.cp" CP pipeline, use:

	python traintest.py ../DefaultOUT_CHARM-like_data.csv ../DefaultOUT_CHARM-like_labels.csv [OUTPUT_PATH] [NB_RUNS] [DISPLAY CONFUSION MATRIX (1:yes, 0:no)] "lda" "kfold" [NB_FOLDS]

	Where [NB_RUNS] is the number of times you would like the training/validation to repeated and [NB_FOLDS] is the number of folds in k-fold cross-validation. [OUTPUT_PATH] is the path to the directory here you would like the program to output the results ("results_summary.txt") and the classifier ("wnd_classifier.pk" or "pcalda_classifier.pk")
	
	* If one wants to save the confusion matrices in addition to the default results_summary.txt (which contains classification accuracies and the list of parameters used to do the experiment), [DISPLAY CONFUSION MATRIX] should be set to 1 and everything should be piped in a text file as in the following example:

	python traintest.py ../DefaultOUT_CHARM-like_data.csv ../DefaultOUT_CHARM-like_labels.csv [OUTPUT_PATH] [NB_RUNS] 1 "lda" "kfold" [NB_FOLDS] > ../confusion_matrices.txt

	* Here's an example of a command that will run 10 rounds of 10-fold cross-validation using PCA-LDA and save the confusion matrices:

	python traintest.py ../DefaultOUT_CHARM-like_data.csv ../DefaultOUT_CHARM-like_labels.csv ../ 10 1 "lda" "kfold" 10 > ../confusion_matrices.txt
	
	Here's one that will run 100 rounds of "save 25%" validation using WND-CHARM without saving confusion matrices:

	python traintest.py ../DefaultOUT_CHARM-like_data.csv ../DefaultOUT_CHARM-like_labels.csv ../ 100 0 "wnd" "save25"
	
 
3) Classifying using classify.py, located in the Classifier folder
	* Use the following command:

	python classify.py [CLASSIFIER_FILE (.pk)] [DATA_FILE (.csv)] [OUTPUT_PATH]

	Where [CLASSIFIER_FILE (.pk)] is the classifier outputted by traintest.py, [DATA_FILE (.csv)] is the output feature file extracted using the "[ClassifyMode]CHARM-like.cp" pipeline, and [OUTPUT_PATH] is the path to the directory where the output "predicted_labels.csv" will be created.


== Notice ==

If images were acquired in several channels, either modify image loading in the "CHARM-like.cp" pipelines to handle multiple channels, or run the pipeline separately on either channels and concatenate the output files afterwards (custom scripts are available on demand for this).
