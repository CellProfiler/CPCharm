CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:11710

LoadImages:[module_num:1|svn_version:\'11587\'|variable_revision_number:11|show_window:False|notes:\x5B\x5D]
    File type to be loaded:individual images
    File selection method:Text-Exact match
    Number of images in each group?:3
    Type the text that the excluded images have in common:Do not use
    Analyze all subfolders within the selected folder?:All
    Input image file location:Default Input Folder\x7CNone
    Check image sets for missing or duplicate files?:Yes
    Group images by metadata?:No
    Exclude certain files?:No
    Specify metadata fields to group by:
    Select subfolders to analyze:
    Image count:1
    Text that these images have in common (case-sensitive):.tif
    Position of this image in each group:1
    Extract metadata from where?:Both
    Regular expression that finds metadata in the file name:(?P<Key>.*)-(?P<HoldOut>\x5BA-Z\x5D)-.*.tif
    Type the regular expression that finds metadata in the subfolder path:.*\x5B\\\\/\x5D(?P<Class>.*)$
    Channel count:1
    Group the movie frames?:No
    Grouping method:Interleaved
    Number of channels per group:3
    Load the input as images or objects?:Images
    Name this loaded image:In
    Name this loaded object:Nuclei
    Retain outlines of loaded objects?:No
    Name the outline image:LoadedImageOutlines
    Channel number:1
    Rescale intensities?:No

ApplyThreshold:[module_num:2|svn_version:\'6746\'|variable_revision_number:5|show_window:False|notes:\x5B\x5D]
    Select the input image:In
    Name the output image:ThreshBlue_Otsu2W
    Select the output image type:Binary (black and white)
    Set pixels below or above the threshold to zero?:Below threshold
    Subtract the threshold value from the remaining pixel intensities?:No
    Number of pixels by which to expand the thresholding around those excluded bright pixels:0.0
    Select the thresholding method:Otsu Global
    Manual threshold:0.0
    Lower and upper bounds on threshold:0.000000,1.000000
    Threshold correction factor:1
    Approximate fraction of image covered by objects?:0.01
    Select the input objects:None
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the measurement to threshold with:None

ApplyThreshold:[module_num:3|svn_version:\'6746\'|variable_revision_number:5|show_window:False|notes:\x5B\x5D]
    Select the input image:In
    Name the output image:ThreshBlue_Otsu3FW
    Select the output image type:Binary (black and white)
    Set pixels below or above the threshold to zero?:Below threshold
    Subtract the threshold value from the remaining pixel intensities?:No
    Number of pixels by which to expand the thresholding around those excluded bright pixels:0.0
    Select the thresholding method:Otsu Global
    Manual threshold:0.0
    Lower and upper bounds on threshold:0.000000,1.000000
    Threshold correction factor:1
    Approximate fraction of image covered by objects?:0.01
    Select the input objects:None
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the measurement to threshold with:None

ApplyThreshold:[module_num:4|svn_version:\'6746\'|variable_revision_number:5|show_window:False|notes:\x5B\x5D]
    Select the input image:In
    Name the output image:ThreshBlue_Otsu3BW
    Select the output image type:Binary (black and white)
    Set pixels below or above the threshold to zero?:Below threshold
    Subtract the threshold value from the remaining pixel intensities?:No
    Number of pixels by which to expand the thresholding around those excluded bright pixels:0.0
    Select the thresholding method:Otsu Global
    Manual threshold:0.0
    Lower and upper bounds on threshold:0.000000,1.000000
    Threshold correction factor:1
    Approximate fraction of image covered by objects?:0.01
    Select the input objects:None
    Two-class or three-class thresholding?:Three classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Background
    Select the measurement to threshold with:None

Transforms:[module_num:5|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Input image name\x3A:In
    Output image name\x3A:InFourier
    Transform choice\x3A:Fourier
    Scale\x3A:3
    Order\x3A:0

Transforms:[module_num:6|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Input image name\x3A:In
    Output image name\x3A:InWavelet
    Transform choice\x3A:Haar Wavelet transform
    Scale\x3A:1
    Order\x3A:0

Transforms:[module_num:7|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Input image name\x3A:In
    Output image name\x3A:InCheby
    Transform choice\x3A:Chebyshev transform
    Scale\x3A:3
    Order\x3A:0

Transforms:[module_num:8|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Input image name\x3A:In
    Output image name\x3A:InChebyStats
    Transform choice\x3A:Chebyshev transform
    Scale\x3A:3
    Order\x3A:20

Transforms:[module_num:9|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Input image name\x3A:InFourier
    Output image name\x3A:InFourierCheby
    Transform choice\x3A:Chebyshev transform
    Scale\x3A:3
    Order\x3A:0

Transforms:[module_num:10|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Input image name\x3A:InFourier
    Output image name\x3A:InFourierWavelet
    Transform choice\x3A:Haar Wavelet transform
    Scale\x3A:1
    Order\x3A:0

Transforms:[module_num:11|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Input image name\x3A:InFourier
    Output image name\x3A:FourierChebyStats
    Transform choice\x3A:Chebyshev transform
    Scale\x3A:3
    Order\x3A:20

EnhanceEdges:[module_num:12|svn_version:\'10300\'|variable_revision_number:2|show_window:False|notes:\x5B\x5D]
    Select the input image:In
    Name the output image:EdgedImage
    Automatically calculate the threshold?:Yes
    Absolute threshold:0.2
    Threshold adjustment factor:1
    Select an edge-finding method:Prewitt
    Select edge direction to enhance:All
    Calculate Gaussian\'s sigma automatically?:Yes
    Gaussian\'s sigma value:10
    Calculate value for low threshold automatically?:Yes
    Low threshold value:0.1

CalculateMoments:[module_num:13|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Hidden:6
    Hidden:0
    Select an image to measure:In
    Select an image to measure:InCheby
    Select an image to measure:InFourier
    Select an image to measure:InFourierCheby
    Select an image to measure:InWavelet
    Select an image to measure:InFourierWavelet
    Moments to compute:Mean,Standard Deviation,Skewness,Kurtosis

CalculateHistogram:[module_num:14|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Hidden:6
    Hidden:0
    Hidden:4
    Select an image to measure:In
    Select an image to measure:InCheby
    Select an image to measure:InFourier
    Select an image to measure:InWavelet
    Select an image to measure:InFourierCheby
    Select an image to measure:InFourierWavelet
    Number of bins:3
    Number of bins:5
    Number of bins:7
    Number of bins:9

CalculateHistogram:[module_num:15|svn_version:\'Unknown\'|variable_revision_number:1|show_window:False|notes:\x5B\x5D]
    Hidden:2
    Hidden:0
    Hidden:1
    Select an image to measure:InChebyStats
    Select an image to measure:FourierChebyStats
    Number of bins:32

MeasureImageQuality:[module_num:16|svn_version:\'11705\'|variable_revision_number:4|show_window:False|notes:\x5B\x5D]
    Calculate metrics for which images?:Select...
    Image count:10
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Scale count:1
    Threshold count:1
    Select the images to measure:EdgedImage
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:In
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:InCheby
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:InFourier
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:InWavelet
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:ThreshBlue_Otsu2W
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:ThreshBlue_Otsu3BW
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:ThreshBlue_Otsu3FW
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:InFourierCheby
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Select the images to measure:InFourierWavelet
    Include the image rescaling value?:Yes
    Calculate blur metrics?:Yes
    Spatial scale for blur measurements:20
    Calculate saturation metrics?:Yes
    Calculate intensity metrics?:Yes
    Calculate thresholds?:Yes
    Use all thresholding methods?:No
    Select a thresholding method:Otsu Global
    Typical fraction of the image covered by objects:0.1
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground

EnhancedMeasureTexture:[module_num:17|svn_version:\'Unknown\'|variable_revision_number:3|show_window:False|notes:\x5B\x5D]
    Hidden:6
    Hidden:0
    Hidden:2
    Select an image to measure:In
    Select an image to measure:InFourier
    Select an image to measure:InWavelet
    Select an image to measure:InCheby
    Select an image to measure:InFourierCheby
    Select an image to measure:InFourierWavelet
    Texture scale to measure:3
    Angles to measure:Horizontal,Vertical,Diagonal,Anti-diagonal
    Texture scale to measure:4
    Angles to measure:Horizontal,Vertical,Diagonal,Anti-diagonal
    Measure Gabor features?:Yes
    Number of angles to compute for Gabor:4
    Measure Tamura features?:Yes
    Features to compute:Coarseness,Contrast,Directionality

ExportToSpreadsheet:[module_num:18|svn_version:\'10880\'|variable_revision_number:7|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder\x7CNone
    Create a GenePattern GCT file?:No
    Select source of sample row name:Metadata
    Select the image to use as the identifier:None
    Select the metadata to use as the identifier:None
    Export all measurements, using default file names?:No
    Press button to select measurements to export:Image\x7CImageQuality_StdIntensity_ThreshBlue_Otsu3BW,Image\x7CImageQuality_StdIntensity_ThreshBlue_Otsu3FW,Image\x7CImageQuality_StdIntensity_ThreshBlue_Otsu2W,Image\x7CImageQuality_StdIntensity_EdgedImage,Image\x7CImageQuality_StdIntensity_InFourierCheby,Image\x7CImageQuality_StdIntensity_InCheby,Image\x7CImageQuality_StdIntensity_InFourierWavelet,Image\x7CImageQuality_StdIntensity_InFourier,Image\x7CImageQuality_StdIntensity_In,Image\x7CImageQuality_StdIntensity_InWavelet,Image\x7CImageQuality_TotalIntensity_ThreshBlue_Otsu3BW,Image\x7CImageQuality_TotalIntensity_ThreshBlue_Otsu3FW,Image\x7CImageQuality_TotalIntensity_ThreshBlue_Otsu2W,Image\x7CImageQuality_TotalIntensity_EdgedImage,Image\x7CImageQuality_TotalIntensity_InFourierCheby,Image\x7CImageQuality_TotalIntensity_InCheby,Image\x7CImageQuality_TotalIntensity_InFourierWavelet,Image\x7CImageQuality_TotalIntensity_InFourier,Image\x7CImageQuality_TotalIntensity_In,Image\x7CImageQuality_TotalIntensity_InWavelet,Image\x7CImageQuality_MeanIntensity_ThreshBlue_Otsu3BW,Image\x7CImageQuality_MeanIntensity_ThreshBlue_Otsu3FW,Image\x7CImageQuality_MeanIntensity_ThreshBlue_Otsu2W,Image\x7CImageQuality_MeanIntensity_EdgedImage,Image\x7CImageQuality_MeanIntensity_InFourierCheby,Image\x7CImageQuality_MeanIntensity_InCheby,Image\x7CImageQuality_MeanIntensity_InFourierWavelet,Image\x7CImageQuality_MeanIntensity_InFourier,Image\x7CImageQuality_MeanIntensity_In,Image\x7CImageQuality_MeanIntensity_InWavelet,Image\x7CImageQuality_PercentMinimal_InFourierCheby,Image\x7CImageQuality_PercentMinimal_InCheby,Image\x7CImageQuality_PercentMinimal_InFourierWavelet,Image\x7CImageQuality_PercentMinimal_InFourier,Image\x7CImageQuality_PercentMinimal_In,Image\x7CImageQuality_PercentMinimal_InWavelet,Image\x7CImageQuality_MaxIntensity_EdgedImage,Image\x7CImageQuality_MaxIntensity_InFourierCheby,Image\x7CImageQuality_MaxIntensity_InCheby,Image\x7CImageQuality_MaxIntensity_InFourierWavelet,Image\x7CImageQuality_MaxIntensity_InFourier,Image\x7CImageQuality_MaxIntensity_In,Image\x7CImageQuality_MaxIntensity_InWavelet,Image\x7CImageQuality_PercentMaximal_InFourierCheby,Image\x7CImageQuality_PercentMaximal_InCheby,Image\x7CImageQuality_PercentMaximal_InFourierWavelet,Image\x7CImageQuality_PercentMaximal_InFourier,Image\x7CImageQuality_PercentMaximal_In,Image\x7CImageQuality_PercentMaximal_InWavelet,Image\x7CTexture_DifferenceEntropy_InFourierCheby_3_0,Image\x7CTexture_DifferenceEntropy_InFourierCheby_3_45,Image\x7CTexture_DifferenceEntropy_InFourierCheby_3_135,Image\x7CTexture_DifferenceEntropy_InFourierCheby_3_90,Image\x7CTexture_DifferenceEntropy_InFourierCheby_4_0,Image\x7CTexture_DifferenceEntropy_InFourierCheby_4_45,Image\x7CTexture_DifferenceEntropy_InFourierCheby_4_135,Image\x7CTexture_DifferenceEntropy_InFourierCheby_4_90,Image\x7CTexture_DifferenceEntropy_InCheby_3_0,Image\x7CTexture_DifferenceEntropy_InCheby_3_45,Image\x7CTexture_DifferenceEntropy_InCheby_3_135,Image\x7CTexture_DifferenceEntropy_InCheby_3_90,Image\x7CTexture_DifferenceEntropy_InCheby_4_0,Image\x7CTexture_DifferenceEntropy_InCheby_4_45,Image\x7CTexture_DifferenceEntropy_InCheby_4_135,Image\x7CTexture_DifferenceEntropy_InCheby_4_90,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_3_0,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_3_45,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_3_135,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_3_90,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_4_0,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_4_45,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_4_135,Image\x7CTexture_DifferenceEntropy_InFourierWavelet_4_90,Image\x7CTexture_DifferenceEntropy_InFourier_3_0,Image\x7CTexture_DifferenceEntropy_InFourier_3_45,Image\x7CTexture_DifferenceEntropy_InFourier_3_135,Image\x7CTexture_DifferenceEntropy_InFourier_3_90,Image\x7CTexture_DifferenceEntropy_InFourier_4_0,Image\x7CTexture_DifferenceEntropy_InFourier_4_45,Image\x7CTexture_DifferenceEntropy_InFourier_4_135,Image\x7CTexture_DifferenceEntropy_InFourier_4_90,Image\x7CTexture_DifferenceEntropy_In_3_0,Image\x7CTexture_DifferenceEntropy_In_3_45,Image\x7CTexture_DifferenceEntropy_In_3_135,Image\x7CTexture_DifferenceEntropy_In_3_90,Image\x7CTexture_DifferenceEntropy_In_4_0,Image\x7CTexture_DifferenceEntropy_In_4_45,Image\x7CTexture_DifferenceEntropy_In_4_135,Image\x7CTexture_DifferenceEntropy_In_4_90,Image\x7CTexture_DifferenceEntropy_InWavelet_3_0,Image\x7CTexture_DifferenceEntropy_InWavelet_3_45,Image\x7CTexture_DifferenceEntropy_InWavelet_3_135,Image\x7CTexture_DifferenceEntropy_InWavelet_3_90,Image\x7CTexture_DifferenceEntropy_InWavelet_4_0,Image\x7CTexture_DifferenceEntropy_InWavelet_4_45,Image\x7CTexture_DifferenceEntropy_InWavelet_4_135,Image\x7CTexture_DifferenceEntropy_InWavelet_4_90,Image\x7CTexture_InfoMeas1_InFourierCheby_3_0,Image\x7CTexture_InfoMeas1_InFourierCheby_3_45,Image\x7CTexture_InfoMeas1_InFourierCheby_3_135,Image\x7CTexture_InfoMeas1_InFourierCheby_3_90,Image\x7CTexture_InfoMeas1_InFourierCheby_4_0,Image\x7CTexture_InfoMeas1_InFourierCheby_4_45,Image\x7CTexture_InfoMeas1_InFourierCheby_4_135,Image\x7CTexture_InfoMeas1_InFourierCheby_4_90,Image\x7CTexture_InfoMeas1_InCheby_3_0,Image\x7CTexture_InfoMeas1_InCheby_3_45,Image\x7CTexture_InfoMeas1_InCheby_3_135,Image\x7CTexture_InfoMeas1_InCheby_3_90,Image\x7CTexture_InfoMeas1_InCheby_4_0,Image\x7CTexture_InfoMeas1_InCheby_4_45,Image\x7CTexture_InfoMeas1_InCheby_4_135,Image\x7CTexture_InfoMeas1_InCheby_4_90,Image\x7CTexture_InfoMeas1_InFourierWavelet_3_0,Image\x7CTexture_InfoMeas1_InFourierWavelet_3_45,Image\x7CTexture_InfoMeas1_InFourierWavelet_3_135,Image\x7CTexture_InfoMeas1_InFourierWavelet_3_90,Image\x7CTexture_InfoMeas1_InFourierWavelet_4_0,Image\x7CTexture_InfoMeas1_InFourierWavelet_4_45,Image\x7CTexture_InfoMeas1_InFourierWavelet_4_135,Image\x7CTexture_InfoMeas1_InFourierWavelet_4_90,Image\x7CTexture_InfoMeas1_InFourier_3_0,Image\x7CTexture_InfoMeas1_InFourier_3_45,Image\x7CTexture_InfoMeas1_InFourier_3_135,Image\x7CTexture_InfoMeas1_InFourier_3_90,Image\x7CTexture_InfoMeas1_InFourier_4_0,Image\x7CTexture_InfoMeas1_InFourier_4_45,Image\x7CTexture_InfoMeas1_InFourier_4_135,Image\x7CTexture_InfoMeas1_InFourier_4_90,Image\x7CTexture_InfoMeas1_In_3_0,Image\x7CTexture_InfoMeas1_In_3_45,Image\x7CTexture_InfoMeas1_In_3_135,Image\x7CTexture_InfoMeas1_In_3_90,Image\x7CTexture_InfoMeas1_In_4_0,Image\x7CTexture_InfoMeas1_In_4_45,Image\x7CTexture_InfoMeas1_In_4_135,Image\x7CTexture_InfoMeas1_In_4_90,Image\x7CTexture_InfoMeas1_InWavelet_3_0,Image\x7CTexture_InfoMeas1_InWavelet_3_45,Image\x7CTexture_InfoMeas1_InWavelet_3_135,Image\x7CTexture_InfoMeas1_InWavelet_3_90,Image\x7CTexture_InfoMeas1_InWavelet_4_0,Image\x7CTexture_InfoMeas1_InWavelet_4_45,Image\x7CTexture_InfoMeas1_InWavelet_4_135,Image\x7CTexture_InfoMeas1_InWavelet_4_90,Image\x7CTexture_DifferenceVariance_InFourierCheby_3_0,Image\x7CTexture_DifferenceVariance_InFourierCheby_3_45,Image\x7CTexture_DifferenceVariance_InFourierCheby_3_135,Image\x7CTexture_DifferenceVariance_InFourierCheby_3_90,Image\x7CTexture_DifferenceVariance_InFourierCheby_4_0,Image\x7CTexture_DifferenceVariance_InFourierCheby_4_45,Image\x7CTexture_DifferenceVariance_InFourierCheby_4_135,Image\x7CTexture_DifferenceVariance_InFourierCheby_4_90,Image\x7CTexture_DifferenceVariance_InCheby_3_0,Image\x7CTexture_DifferenceVariance_InCheby_3_45,Image\x7CTexture_DifferenceVariance_InCheby_3_135,Image\x7CTexture_DifferenceVariance_InCheby_3_90,Image\x7CTexture_DifferenceVariance_InCheby_4_0,Image\x7CTexture_DifferenceVariance_InCheby_4_45,Image\x7CTexture_DifferenceVariance_InCheby_4_135,Image\x7CTexture_DifferenceVariance_InCheby_4_90,Image\x7CTexture_DifferenceVariance_InFourierWavelet_3_0,Image\x7CTexture_DifferenceVariance_InFourierWavelet_3_45,Image\x7CTexture_DifferenceVariance_InFourierWavelet_3_135,Image\x7CTexture_DifferenceVariance_InFourierWavelet_3_90,Image\x7CTexture_DifferenceVariance_InFourierWavelet_4_0,Image\x7CTexture_DifferenceVariance_InFourierWavelet_4_45,Image\x7CTexture_DifferenceVariance_InFourierWavelet_4_135,Image\x7CTexture_DifferenceVariance_InFourierWavelet_4_90,Image\x7CTexture_DifferenceVariance_InFourier_3_0,Image\x7CTexture_DifferenceVariance_InFourier_3_45,Image\x7CTexture_DifferenceVariance_InFourier_3_135,Image\x7CTexture_DifferenceVariance_InFourier_3_90,Image\x7CTexture_DifferenceVariance_InFourier_4_0,Image\x7CTexture_DifferenceVariance_InFourier_4_45,Image\x7CTexture_DifferenceVariance_InFourier_4_135,Image\x7CTexture_DifferenceVariance_InFourier_4_90,Image\x7CTexture_DifferenceVariance_In_3_0,Image\x7CTexture_DifferenceVariance_In_3_45,Image\x7CTexture_DifferenceVariance_In_3_135,Image\x7CTexture_DifferenceVariance_In_3_90,Image\x7CTexture_DifferenceVariance_In_4_0,Image\x7CTexture_DifferenceVariance_In_4_45,Image\x7CTexture_DifferenceVariance_In_4_135,Image\x7CTexture_DifferenceVariance_In_4_90,Image\x7CTexture_DifferenceVariance_InWavelet_3_0,Image\x7CTexture_DifferenceVariance_InWavelet_3_45,Image\x7CTexture_DifferenceVariance_InWavelet_3_135,Image\x7CTexture_DifferenceVariance_InWavelet_3_90,Image\x7CTexture_DifferenceVariance_InWavelet_4_0,Image\x7CTexture_DifferenceVariance_InWavelet_4_45,Image\x7CTexture_DifferenceVariance_InWavelet_4_135,Image\x7CTexture_DifferenceVariance_InWavelet_4_90,Image\x7CTexture_SumVariance_InFourierCheby_3_0,Image\x7CTexture_SumVariance_InFourierCheby_3_45,Image\x7CTexture_SumVariance_InFourierCheby_3_135,Image\x7CTexture_SumVariance_InFourierCheby_3_90,Image\x7CTexture_SumVariance_InFourierCheby_4_0,Image\x7CTexture_SumVariance_InFourierCheby_4_45,Image\x7CTexture_SumVariance_InFourierCheby_4_135,Image\x7CTexture_SumVariance_InFourierCheby_4_90,Image\x7CTexture_SumVariance_InCheby_3_0,Image\x7CTexture_SumVariance_InCheby_3_45,Image\x7CTexture_SumVariance_InCheby_3_135,Image\x7CTexture_SumVariance_InCheby_3_90,Image\x7CTexture_SumVariance_InCheby_4_0,Image\x7CTexture_SumVariance_InCheby_4_45,Image\x7CTexture_SumVariance_InCheby_4_135,Image\x7CTexture_SumVariance_InCheby_4_90,Image\x7CTexture_SumVariance_InFourierWavelet_3_0,Image\x7CTexture_SumVariance_InFourierWavelet_3_45,Image\x7CTexture_SumVariance_InFourierWavelet_3_135,Image\x7CTexture_SumVariance_InFourierWavelet_3_90,Image\x7CTexture_SumVariance_InFourierWavelet_4_0,Image\x7CTexture_SumVariance_InFourierWavelet_4_45,Image\x7CTexture_SumVariance_InFourierWavelet_4_135,Image\x7CTexture_SumVariance_InFourierWavelet_4_90,Image\x7CTexture_SumVariance_InFourier_3_0,Image\x7CTexture_SumVariance_InFourier_3_45,Image\x7CTexture_SumVariance_InFourier_3_135,Image\x7CTexture_SumVariance_InFourier_3_90,Image\x7CTexture_SumVariance_InFourier_4_0,Image\x7CTexture_SumVariance_InFourier_4_45,Image\x7CTexture_SumVariance_InFourier_4_135,Image\x7CTexture_SumVariance_InFourier_4_90,Image\x7CTexture_SumVariance_In_3_0,Image\x7CTexture_SumVariance_In_3_45,Image\x7CTexture_SumVariance_In_3_135,Image\x7CTexture_SumVariance_In_3_90,Image\x7CTexture_SumVariance_In_4_0,Image\x7CTexture_SumVariance_In_4_45,Image\x7CTexture_SumVariance_In_4_135,Image\x7CTexture_SumVariance_In_4_90,Image\x7CTexture_SumVariance_InWavelet_3_0,Image\x7CTexture_SumVariance_InWavelet_3_45,Image\x7CTexture_SumVariance_InWavelet_3_135,Image\x7CTexture_SumVariance_InWavelet_3_90,Image\x7CTexture_SumVariance_InWavelet_4_0,Image\x7CTexture_SumVariance_InWavelet_4_45,Image\x7CTexture_SumVariance_InWavelet_4_135,Image\x7CTexture_SumVariance_InWavelet_4_90,Image\x7CTexture_Gabor_InFourierCheby_3,Image\x7CTexture_Gabor_InFourierCheby_4,Image\x7CTexture_Gabor_InCheby_3,Image\x7CTexture_Gabor_InCheby_4,Image\x7CTexture_Gabor_InFourierWavelet_3,Image\x7CTexture_Gabor_InFourierWavelet_4,Image\x7CTexture_Gabor_InFourier_3,Image\x7CTexture_Gabor_InFourier_4,Image\x7CTexture_Gabor_In_3,Image\x7CTexture_Gabor_In_4,Image\x7CTexture_Gabor_InWavelet_3,Image\x7CTexture_Gabor_InWavelet_4,Image\x7CTexture_AngularSecondMoment_InFourierCheby_3_0,Image\x7CTexture_AngularSecondMoment_InFourierCheby_3_45,Image\x7CTexture_AngularSecondMoment_InFourierCheby_3_135,Image\x7CTexture_AngularSecondMoment_InFourierCheby_3_90,Image\x7CTexture_AngularSecondMoment_InFourierCheby_4_0,Image\x7CTexture_AngularSecondMoment_InFourierCheby_4_45,Image\x7CTexture_AngularSecondMoment_InFourierCheby_4_135,Image\x7CTexture_AngularSecondMoment_InFourierCheby_4_90,Image\x7CTexture_AngularSecondMoment_InCheby_3_0,Image\x7CTexture_AngularSecondMoment_InCheby_3_45,Image\x7CTexture_AngularSecondMoment_InCheby_3_135,Image\x7CTexture_AngularSecondMoment_InCheby_3_90,Image\x7CTexture_AngularSecondMoment_InCheby_4_0,Image\x7CTexture_AngularSecondMoment_InCheby_4_45,Image\x7CTexture_AngularSecondMoment_InCheby_4_135,Image\x7CTexture_AngularSecondMoment_InCheby_4_90,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_3_0,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_3_45,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_3_135,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_3_90,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_4_0,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_4_45,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_4_135,Image\x7CTexture_AngularSecondMoment_InFourierWavelet_4_90,Image\x7CTexture_AngularSecondMoment_InFourier_3_0,Image\x7CTexture_AngularSecondMoment_InFourier_3_45,Image\x7CTexture_AngularSecondMoment_InFourier_3_135,Image\x7CTexture_AngularSecondMoment_InFourier_3_90,Image\x7CTexture_AngularSecondMoment_InFourier_4_0,Image\x7CTexture_AngularSecondMoment_InFourier_4_45,Image\x7CTexture_AngularSecondMoment_InFourier_4_135,Image\x7CTexture_AngularSecondMoment_InFourier_4_90,Image\x7CTexture_AngularSecondMoment_In_3_0,Image\x7CTexture_AngularSecondMoment_In_3_45,Image\x7CTexture_AngularSecondMoment_In_3_135,Image\x7CTexture_AngularSecondMoment_In_3_90,Image\x7CTexture_AngularSecondMoment_In_4_0,Image\x7CTexture_AngularSecondMoment_In_4_45,Image\x7CTexture_AngularSecondMoment_In_4_135,Image\x7CTexture_AngularSecondMoment_In_4_90,Image\x7CTexture_AngularSecondMoment_InWavelet_3_0,Image\x7CTexture_AngularSecondMoment_InWavelet_3_45,Image\x7CTexture_AngularSecondMoment_InWavelet_3_135,Image\x7CTexture_AngularSecondMoment_InWavelet_3_90,Image\x7CTexture_AngularSecondMoment_InWavelet_4_0,Image\x7CTexture_AngularSecondMoment_InWavelet_4_45,Image\x7CTexture_AngularSecondMoment_InWavelet_4_135,Image\x7CTexture_AngularSecondMoment_InWavelet_4_90,Image\x7CTexture_Tamura_Coarseness_InFourierCheby,Image\x7CTexture_Tamura_Coarseness_InCheby,Image\x7CTexture_Tamura_Coarseness_InFourierWavelet,Image\x7CTexture_Tamura_Coarseness_InFourier,Image\x7CTexture_Tamura_Coarseness_In,Image\x7CTexture_Tamura_Coarseness_InWavelet,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin1_InFourierCheby,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin1_InCheby,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin1_InFourierWavelet,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin1_InFourier,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin1_In,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin1_InWavelet,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin0_InFourierCheby,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin0_InCheby,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin0_InFourierWavelet,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin0_InFourier,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin0_In,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin0_InWavelet,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin2_InFourierCheby,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin2_InCheby,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin2_InFourierWavelet,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin2_InFourier,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin2_In,Image\x7CTexture_Tamura_CoarsenessHist_3BinsHist_Bin2_InWavelet,Image\x7CTexture_Tamura_Directionality_InFourierCheby,Image\x7CTexture_Tamura_Directionality_InCheby,Image\x7CTexture_Tamura_Directionality_InFourierWavelet,Image\x7CTexture_Tamura_Directionality_InFourier,Image\x7CTexture_Tamura_Directionality_In,Image\x7CTexture_Tamura_Directionality_InWavelet,Image\x7CTexture_Tamura_Contrast_InFourierCheby,Image\x7CTexture_Tamura_Contrast_InCheby,Image\x7CTexture_Tamura_Contrast_InFourierWavelet,Image\x7CTexture_Tamura_Contrast_InFourier,Image\x7CTexture_Tamura_Contrast_In,Image\x7CTexture_Tamura_Contrast_InWavelet,Image\x7CTexture_Entropy_InFourierCheby_3_0,Image\x7CTexture_Entropy_InFourierCheby_3_45,Image\x7CTexture_Entropy_InFourierCheby_3_135,Image\x7CTexture_Entropy_InFourierCheby_3_90,Image\x7CTexture_Entropy_InFourierCheby_4_0,Image\x7CTexture_Entropy_InFourierCheby_4_45,Image\x7CTexture_Entropy_InFourierCheby_4_135,Image\x7CTexture_Entropy_InFourierCheby_4_90,Image\x7CTexture_Entropy_InCheby_3_0,Image\x7CTexture_Entropy_InCheby_3_45,Image\x7CTexture_Entropy_InCheby_3_135,Image\x7CTexture_Entropy_InCheby_3_90,Image\x7CTexture_Entropy_InCheby_4_0,Image\x7CTexture_Entropy_InCheby_4_45,Image\x7CTexture_Entropy_InCheby_4_135,Image\x7CTexture_Entropy_InCheby_4_90,Image\x7CTexture_Entropy_InFourierWavelet_3_0,Image\x7CTexture_Entropy_InFourierWavelet_3_45,Image\x7CTexture_Entropy_InFourierWavelet_3_135,Image\x7CTexture_Entropy_InFourierWavelet_3_90,Image\x7CTexture_Entropy_InFourierWavelet_4_0,Image\x7CTexture_Entropy_InFourierWavelet_4_45,Image\x7CTexture_Entropy_InFourierWavelet_4_135,Image\x7CTexture_Entropy_InFourierWavelet_4_90,Image\x7CTexture_Entropy_InFourier_3_0,Image\x7CTexture_Entropy_InFourier_3_45,Image\x7CTexture_Entropy_InFourier_3_135,Image\x7CTexture_Entropy_InFourier_3_90,Image\x7CTexture_Entropy_InFourier_4_0,Image\x7CTexture_Entropy_InFourier_4_45,Image\x7CTexture_Entropy_InFourier_4_135,Image\x7CTexture_Entropy_InFourier_4_90,Image\x7CTexture_Entropy_In_3_0,Image\x7CTexture_Entropy_In_3_45,Image\x7CTexture_Entropy_In_3_135,Image\x7CTexture_Entropy_In_3_90,Image\x7CTexture_Entropy_In_4_0,Image\x7CTexture_Entropy_In_4_45,Image\x7CTexture_Entropy_In_4_135,Image\x7CTexture_Entropy_In_4_90,Image\x7CTexture_Entropy_InWavelet_3_0,Image\x7CTexture_Entropy_InWavelet_3_45,Image\x7CTexture_Entropy_InWavelet_3_135,Image\x7CTexture_Entropy_InWavelet_3_90,Image\x7CTexture_Entropy_InWavelet_4_0,Image\x7CTexture_Entropy_InWavelet_4_45,Image\x7CTexture_Entropy_InWavelet_4_135,Image\x7CTexture_Entropy_InWavelet_4_90,Image\x7CTexture_Correlation_InFourierCheby_3_0,Image\x7CTexture_Correlation_InFourierCheby_3_45,Image\x7CTexture_Correlation_InFourierCheby_3_135,Image\x7CTexture_Correlation_InFourierCheby_3_90,Image\x7CTexture_Correlation_InFourierCheby_4_0,Image\x7CTexture_Correlation_InFourierCheby_4_45,Image\x7CTexture_Correlation_InFourierCheby_4_135,Image\x7CTexture_Correlation_InFourierCheby_4_90,Image\x7CTexture_Correlation_InCheby_3_0,Image\x7CTexture_Correlation_InCheby_3_45,Image\x7CTexture_Correlation_InCheby_3_135,Image\x7CTexture_Correlation_InCheby_3_90,Image\x7CTexture_Correlation_InCheby_4_0,Image\x7CTexture_Correlation_InCheby_4_45,Image\x7CTexture_Correlation_InCheby_4_135,Image\x7CTexture_Correlation_InCheby_4_90,Image\x7CTexture_Correlation_InFourierWavelet_3_0,Image\x7CTexture_Correlation_InFourierWavelet_3_45,Image\x7CTexture_Correlation_InFourierWavelet_3_135,Image\x7CTexture_Correlation_InFourierWavelet_3_90,Image\x7CTexture_Correlation_InFourierWavelet_4_0,Image\x7CTexture_Correlation_InFourierWavelet_4_45,Image\x7CTexture_Correlation_InFourierWavelet_4_135,Image\x7CTexture_Correlation_InFourierWavelet_4_90,Image\x7CTexture_Correlation_InFourier_3_0,Image\x7CTexture_Correlation_InFourier_3_45,Image\x7CTexture_Correlation_InFourier_3_135,Image\x7CTexture_Correlation_InFourier_3_90,Image\x7CTexture_Correlation_InFourier_4_0,Image\x7CTexture_Correlation_InFourier_4_45,Image\x7CTexture_Correlation_InFourier_4_135,Image\x7CTexture_Correlation_InFourier_4_90,Image\x7CTexture_Correlation_In_3_0,Image\x7CTexture_Correlation_In_3_45,Image\x7CTexture_Correlation_In_3_135,Image\x7CTexture_Correlation_In_3_90,Image\x7CTexture_Correlation_In_4_0,Image\x7CTexture_Correlation_In_4_45,Image\x7CTexture_Correlation_In_4_135,Image\x7CTexture_Correlation_In_4_90,Image\x7CTexture_Correlation_InWavelet_3_0,Image\x7CTexture_Correlation_InWavelet_3_45,Image\x7CTexture_Correlation_InWavelet_3_135,Image\x7CTexture_Correlation_InWavelet_3_90,Image\x7CTexture_Correlation_InWavelet_4_0,Image\x7CTexture_Correlation_InWavelet_4_45,Image\x7CTexture_Correlation_InWavelet_4_135,Image\x7CTexture_Correlation_InWavelet_4_90,Image\x7CTexture_SumAverage_InFourierCheby_3_0,Image\x7CTexture_SumAverage_InFourierCheby_3_45,Image\x7CTexture_SumAverage_InFourierCheby_3_135,Image\x7CTexture_SumAverage_InFourierCheby_3_90,Image\x7CTexture_SumAverage_InFourierCheby_4_0,Image\x7CTexture_SumAverage_InFourierCheby_4_45,Image\x7CTexture_SumAverage_InFourierCheby_4_135,Image\x7CTexture_SumAverage_InFourierCheby_4_90,Image\x7CTexture_SumAverage_InCheby_3_0,Image\x7CTexture_SumAverage_InCheby_3_45,Image\x7CTexture_SumAverage_InCheby_3_135,Image\x7CTexture_SumAverage_InCheby_3_90,Image\x7CTexture_SumAverage_InCheby_4_0,Image\x7CTexture_SumAverage_InCheby_4_45,Image\x7CTexture_SumAverage_InCheby_4_135,Image\x7CTexture_SumAverage_InCheby_4_90,Image\x7CTexture_SumAverage_InFourierWavelet_3_0,Image\x7CTexture_SumAverage_InFourierWavelet_3_45,Image\x7CTexture_SumAverage_InFourierWavelet_3_135,Image\x7CTexture_SumAverage_InFourierWavelet_3_90,Image\x7CTexture_SumAverage_InFourierWavelet_4_0,Image\x7CTexture_SumAverage_InFourierWavelet_4_45,Image\x7CTexture_SumAverage_InFourierWavelet_4_135,Image\x7CTexture_SumAverage_InFourierWavelet_4_90,Image\x7CTexture_SumAverage_InFourier_3_0,Image\x7CTexture_SumAverage_InFourier_3_45,Image\x7CTexture_SumAverage_InFourier_3_135,Image\x7CTexture_SumAverage_InFourier_3_90,Image\x7CTexture_SumAverage_InFourier_4_0,Image\x7CTexture_SumAverage_InFourier_4_45,Image\x7CTexture_SumAverage_InFourier_4_135,Image\x7CTexture_SumAverage_InFourier_4_90,Image\x7CTexture_SumAverage_In_3_0,Image\x7CTexture_SumAverage_In_3_45,Image\x7CTexture_SumAverage_In_3_135,Image\x7CTexture_SumAverage_In_3_90,Image\x7CTexture_SumAverage_In_4_0,Image\x7CTexture_SumAverage_In_4_45,Image\x7CTexture_SumAverage_In_4_135,Image\x7CTexture_SumAverage_In_4_90,Image\x7CTexture_SumAverage_InWavelet_3_0,Image\x7CTexture_SumAverage_InWavelet_3_45,Image\x7CTexture_SumAverage_InWavelet_3_135,Image\x7CTexture_SumAverage_InWavelet_3_90,Image\x7CTexture_SumAverage_InWavelet_4_0,Image\x7CTexture_SumAverage_InWavelet_4_45,Image\x7CTexture_SumAverage_InWavelet_4_135,Image\x7CTexture_SumAverage_InWavelet_4_90,Image\x7CTexture_Variance_InFourierCheby_3_0,Image\x7CTexture_Variance_InFourierCheby_3_45,Image\x7CTexture_Variance_InFourierCheby_3_135,Image\x7CTexture_Variance_InFourierCheby_3_90,Image\x7CTexture_Variance_InFourierCheby_4_0,Image\x7CTexture_Variance_InFourierCheby_4_45,Image\x7CTexture_Variance_InFourierCheby_4_135,Image\x7CTexture_Variance_InFourierCheby_4_90,Image\x7CTexture_Variance_InCheby_3_0,Image\x7CTexture_Variance_InCheby_3_45,Image\x7CTexture_Variance_InCheby_3_135,Image\x7CTexture_Variance_InCheby_3_90,Image\x7CTexture_Variance_InCheby_4_0,Image\x7CTexture_Variance_InCheby_4_45,Image\x7CTexture_Variance_InCheby_4_135,Image\x7CTexture_Variance_InCheby_4_90,Image\x7CTexture_Variance_InFourierWavelet_3_0,Image\x7CTexture_Variance_InFourierWavelet_3_45,Image\x7CTexture_Variance_InFourierWavelet_3_135,Image\x7CTexture_Variance_InFourierWavelet_3_90,Image\x7CTexture_Variance_InFourierWavelet_4_0,Image\x7CTexture_Variance_InFourierWavelet_4_45,Image\x7CTexture_Variance_InFourierWavelet_4_135,Image\x7CTexture_Variance_InFourierWavelet_4_90,Image\x7CTexture_Variance_InFourier_3_0,Image\x7CTexture_Variance_InFourier_3_45,Image\x7CTexture_Variance_InFourier_3_135,Image\x7CTexture_Variance_InFourier_3_90,Image\x7CTexture_Variance_InFourier_4_0,Image\x7CTexture_Variance_InFourier_4_45,Image\x7CTexture_Variance_InFourier_4_135,Image\x7CTexture_Variance_InFourier_4_90,Image\x7CTexture_Variance_In_3_0,Image\x7CTexture_Variance_In_3_45,Image\x7CTexture_Variance_In_3_135,Image\x7CTexture_Variance_In_3_90,Image\x7CTexture_Variance_In_4_0,Image\x7CTexture_Variance_In_4_45,Image\x7CTexture_Variance_In_4_135,Image\x7CTexture_Variance_In_4_90,Image\x7CTexture_Variance_InWavelet_3_0,Image\x7CTexture_Variance_InWavelet_3_45,Image\x7CTexture_Variance_InWavelet_3_135,Image\x7CTexture_Variance_InWavelet_3_90,Image\x7CTexture_Variance_InWavelet_4_0,Image\x7CTexture_Variance_InWavelet_4_45,Image\x7CTexture_Variance_InWavelet_4_135,Image\x7CTexture_Variance_InWavelet_4_90,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_3_0,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_3_45,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_3_135,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_3_90,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_4_0,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_4_45,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_4_135,Image\x7CTexture_InverseDifferenceMoment_InFourierCheby_4_90,Image\x7CTexture_InverseDifferenceMoment_InCheby_3_0,Image\x7CTexture_InverseDifferenceMoment_InCheby_3_45,Image\x7CTexture_InverseDifferenceMoment_InCheby_3_135,Image\x7CTexture_InverseDifferenceMoment_InCheby_3_90,Image\x7CTexture_InverseDifferenceMoment_InCheby_4_0,Image\x7CTexture_InverseDifferenceMoment_InCheby_4_45,Image\x7CTexture_InverseDifferenceMoment_InCheby_4_135,Image\x7CTexture_InverseDifferenceMoment_InCheby_4_90,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_3_0,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_3_45,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_3_135,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_3_90,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_4_0,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_4_45,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_4_135,Image\x7CTexture_InverseDifferenceMoment_InFourierWavelet_4_90,Image\x7CTexture_InverseDifferenceMoment_InFourier_3_0,Image\x7CTexture_InverseDifferenceMoment_InFourier_3_45,Image\x7CTexture_InverseDifferenceMoment_InFourier_3_135,Image\x7CTexture_InverseDifferenceMoment_InFourier_3_90,Image\x7CTexture_InverseDifferenceMoment_InFourier_4_0,Image\x7CTexture_InverseDifferenceMoment_InFourier_4_45,Image\x7CTexture_InverseDifferenceMoment_InFourier_4_135,Image\x7CTexture_InverseDifferenceMoment_InFourier_4_90,Image\x7CTexture_InverseDifferenceMoment_In_3_0,Image\x7CTexture_InverseDifferenceMoment_In_3_45,Image\x7CTexture_InverseDifferenceMoment_In_3_135,Image\x7CTexture_InverseDifferenceMoment_In_3_90,Image\x7CTexture_InverseDifferenceMoment_In_4_0,Image\x7CTexture_InverseDifferenceMoment_In_4_45,Image\x7CTexture_InverseDifferenceMoment_In_4_135,Image\x7CTexture_InverseDifferenceMoment_In_4_90,Image\x7CTexture_InverseDifferenceMoment_InWavelet_3_0,Image\x7CTexture_InverseDifferenceMoment_InWavelet_3_45,Image\x7CTexture_InverseDifferenceMoment_InWavelet_3_135,Image\x7CTexture_InverseDifferenceMoment_InWavelet_3_90,Image\x7CTexture_InverseDifferenceMoment_InWavelet_4_0,Image\x7CTexture_InverseDifferenceMoment_InWavelet_4_45,Image\x7CTexture_InverseDifferenceMoment_InWavelet_4_135,Image\x7CTexture_InverseDifferenceMoment_InWavelet_4_90,Image\x7CTexture_SumEntropy_InFourierCheby_3_0,Image\x7CTexture_SumEntropy_InFourierCheby_3_45,Image\x7CTexture_SumEntropy_InFourierCheby_3_135,Image\x7CTexture_SumEntropy_InFourierCheby_3_90,Image\x7CTexture_SumEntropy_InFourierCheby_4_0,Image\x7CTexture_SumEntropy_InFourierCheby_4_45,Image\x7CTexture_SumEntropy_InFourierCheby_4_135,Image\x7CTexture_SumEntropy_InFourierCheby_4_90,Image\x7CTexture_SumEntropy_InCheby_3_0,Image\x7CTexture_SumEntropy_InCheby_3_45,Image\x7CTexture_SumEntropy_InCheby_3_135,Image\x7CTexture_SumEntropy_InCheby_3_90,Image\x7CTexture_SumEntropy_InCheby_4_0,Image\x7CTexture_SumEntropy_InCheby_4_45,Image\x7CTexture_SumEntropy_InCheby_4_135,Image\x7CTexture_SumEntropy_InCheby_4_90,Image\x7CTexture_SumEntropy_InFourierWavelet_3_0,Image\x7CTexture_SumEntropy_InFourierWavelet_3_45,Image\x7CTexture_SumEntropy_InFourierWavelet_3_135,Image\x7CTexture_SumEntropy_InFourierWavelet_3_90,Image\x7CTexture_SumEntropy_InFourierWavelet_4_0,Image\x7CTexture_SumEntropy_InFourierWavelet_4_45,Image\x7CTexture_SumEntropy_InFourierWavelet_4_135,Image\x7CTexture_SumEntropy_InFourierWavelet_4_90,Image\x7CTexture_SumEntropy_InFourier_3_0,Image\x7CTexture_SumEntropy_InFourier_3_45,Image\x7CTexture_SumEntropy_InFourier_3_135,Image\x7CTexture_SumEntropy_InFourier_3_90,Image\x7CTexture_SumEntropy_InFourier_4_0,Image\x7CTexture_SumEntropy_InFourier_4_45,Image\x7CTexture_SumEntropy_InFourier_4_135,Image\x7CTexture_SumEntropy_InFourier_4_90,Image\x7CTexture_SumEntropy_In_3_0,Image\x7CTexture_SumEntropy_In_3_45,Image\x7CTexture_SumEntropy_In_3_135,Image\x7CTexture_SumEntropy_In_3_90,Image\x7CTexture_SumEntropy_In_4_0,Image\x7CTexture_SumEntropy_In_4_45,Image\x7CTexture_SumEntropy_In_4_135,Image\x7CTexture_SumEntropy_In_4_90,Image\x7CTexture_SumEntropy_InWavelet_3_0,Image\x7CTexture_SumEntropy_InWavelet_3_45,Image\x7CTexture_SumEntropy_InWavelet_3_135,Image\x7CTexture_SumEntropy_InWavelet_3_90,Image\x7CTexture_SumEntropy_InWavelet_4_0,Image\x7CTexture_SumEntropy_InWavelet_4_45,Image\x7CTexture_SumEntropy_InWavelet_4_135,Image\x7CTexture_SumEntropy_InWavelet_4_90,Image\x7CTexture_Contrast_InFourierCheby_3_0,Image\x7CTexture_Contrast_InFourierCheby_3_45,Image\x7CTexture_Contrast_InFourierCheby_3_135,Image\x7CTexture_Contrast_InFourierCheby_3_90,Image\x7CTexture_Contrast_InFourierCheby_4_0,Image\x7CTexture_Contrast_InFourierCheby_4_45,Image\x7CTexture_Contrast_InFourierCheby_4_135,Image\x7CTexture_Contrast_InFourierCheby_4_90,Image\x7CTexture_Contrast_InCheby_3_0,Image\x7CTexture_Contrast_InCheby_3_45,Image\x7CTexture_Contrast_InCheby_3_135,Image\x7CTexture_Contrast_InCheby_3_90,Image\x7CTexture_Contrast_InCheby_4_0,Image\x7CTexture_Contrast_InCheby_4_45,Image\x7CTexture_Contrast_InCheby_4_135,Image\x7CTexture_Contrast_InCheby_4_90,Image\x7CTexture_Contrast_InFourierWavelet_3_0,Image\x7CTexture_Contrast_InFourierWavelet_3_45,Image\x7CTexture_Contrast_InFourierWavelet_3_135,Image\x7CTexture_Contrast_InFourierWavelet_3_90,Image\x7CTexture_Contrast_InFourierWavelet_4_0,Image\x7CTexture_Contrast_InFourierWavelet_4_45,Image\x7CTexture_Contrast_InFourierWavelet_4_135,Image\x7CTexture_Contrast_InFourierWavelet_4_90,Image\x7CTexture_Contrast_InFourier_3_0,Image\x7CTexture_Contrast_InFourier_3_45,Image\x7CTexture_Contrast_InFourier_3_135,Image\x7CTexture_Contrast_InFourier_3_90,Image\x7CTexture_Contrast_InFourier_4_0,Image\x7CTexture_Contrast_InFourier_4_45,Image\x7CTexture_Contrast_InFourier_4_135,Image\x7CTexture_Contrast_InFourier_4_90,Image\x7CTexture_Contrast_In_3_0,Image\x7CTexture_Contrast_In_3_45,Image\x7CTexture_Contrast_In_3_135,Image\x7CTexture_Contrast_In_3_90,Image\x7CTexture_Contrast_In_4_0,Image\x7CTexture_Contrast_In_4_45,Image\x7CTexture_Contrast_In_4_135,Image\x7CTexture_Contrast_In_4_90,Image\x7CTexture_Contrast_InWavelet_3_0,Image\x7CTexture_Contrast_InWavelet_3_45,Image\x7CTexture_Contrast_InWavelet_3_135,Image\x7CTexture_Contrast_InWavelet_3_90,Image\x7CTexture_Contrast_InWavelet_4_0,Image\x7CTexture_Contrast_InWavelet_4_45,Image\x7CTexture_Contrast_InWavelet_4_135,Image\x7CTexture_Contrast_InWavelet_4_90,Image\x7CTexture_InfoMeas2_InFourierCheby_3_0,Image\x7CTexture_InfoMeas2_InFourierCheby_3_45,Image\x7CTexture_InfoMeas2_InFourierCheby_3_135,Image\x7CTexture_InfoMeas2_InFourierCheby_3_90,Image\x7CTexture_InfoMeas2_InFourierCheby_4_0,Image\x7CTexture_InfoMeas2_InFourierCheby_4_45,Image\x7CTexture_InfoMeas2_InFourierCheby_4_135,Image\x7CTexture_InfoMeas2_InFourierCheby_4_90,Image\x7CTexture_InfoMeas2_InCheby_3_0,Image\x7CTexture_InfoMeas2_InCheby_3_45,Image\x7CTexture_InfoMeas2_InCheby_3_135,Image\x7CTexture_InfoMeas2_InCheby_3_90,Image\x7CTexture_InfoMeas2_InCheby_4_0,Image\x7CTexture_InfoMeas2_InCheby_4_45,Image\x7CTexture_InfoMeas2_InCheby_4_135,Image\x7CTexture_InfoMeas2_InCheby_4_90,Image\x7CTexture_InfoMeas2_InFourierWavelet_3_0,Image\x7CTexture_InfoMeas2_InFourierWavelet_3_45,Image\x7CTexture_InfoMeas2_InFourierWavelet_3_135,Image\x7CTexture_InfoMeas2_InFourierWavelet_3_90,Image\x7CTexture_InfoMeas2_InFourierWavelet_4_0,Image\x7CTexture_InfoMeas2_InFourierWavelet_4_45,Image\x7CTexture_InfoMeas2_InFourierWavelet_4_135,Image\x7CTexture_InfoMeas2_InFourierWavelet_4_90,Image\x7CTexture_InfoMeas2_InFourier_3_0,Image\x7CTexture_InfoMeas2_InFourier_3_45,Image\x7CTexture_InfoMeas2_InFourier_3_135,Image\x7CTexture_InfoMeas2_InFourier_3_90,Image\x7CTexture_InfoMeas2_InFourier_4_0,Image\x7CTexture_InfoMeas2_InFourier_4_45,Image\x7CTexture_InfoMeas2_InFourier_4_135,Image\x7CTexture_InfoMeas2_InFourier_4_90,Image\x7CTexture_InfoMeas2_In_3_0,Image\x7CTexture_InfoMeas2_In_3_45,Image\x7CTexture_InfoMeas2_In_3_135,Image\x7CTexture_InfoMeas2_In_3_90,Image\x7CTexture_InfoMeas2_In_4_0,Image\x7CTexture_InfoMeas2_In_4_45,Image\x7CTexture_InfoMeas2_In_4_135,Image\x7CTexture_InfoMeas2_In_4_90,Image\x7CTexture_InfoMeas2_InWavelet_3_0,Image\x7CTexture_InfoMeas2_InWavelet_3_45,Image\x7CTexture_InfoMeas2_InWavelet_3_135,Image\x7CTexture_InfoMeas2_InWavelet_3_90,Image\x7CTexture_InfoMeas2_InWavelet_4_0,Image\x7CTexture_InfoMeas2_InWavelet_4_45,Image\x7CTexture_InfoMeas2_InWavelet_4_135,Image\x7CTexture_InfoMeas2_InWavelet_4_90,Image\x7CHistogram_32BinsHistBin10_InChebyStats,Image\x7CHistogram_32BinsHistBin10_FourierChebyStats,Image\x7CHistogram_32BinsHistBin19_InChebyStats,Image\x7CHistogram_32BinsHistBin19_FourierChebyStats,Image\x7CHistogram_3BinsHistBin1_InFourierCheby,Image\x7CHistogram_3BinsHistBin1_InCheby,Image\x7CHistogram_3BinsHistBin1_InFourierWavelet,Image\x7CHistogram_3BinsHistBin1_InFourier,Image\x7CHistogram_3BinsHistBin1_In,Image\x7CHistogram_3BinsHistBin1_InWavelet,Image\x7CHistogram_3BinsHistBin0_InFourierCheby,Image\x7CHistogram_3BinsHistBin0_InCheby,Image\x7CHistogram_3BinsHistBin0_InFourierWavelet,Image\x7CHistogram_3BinsHistBin0_InFourier,Image\x7CHistogram_3BinsHistBin0_In,Image\x7CHistogram_3BinsHistBin0_InWavelet,Image\x7CHistogram_3BinsHistBin2_InFourierCheby,Image\x7CHistogram_3BinsHistBin2_InCheby,Image\x7CHistogram_3BinsHistBin2_InFourierWavelet,Image\x7CHistogram_3BinsHistBin2_InFourier,Image\x7CHistogram_3BinsHistBin2_In,Image\x7CHistogram_3BinsHistBin2_InWavelet,Image\x7CHistogram_32BinsHistBin11_InChebyStats,Image\x7CHistogram_32BinsHistBin11_FourierChebyStats,Image\x7CHistogram_32BinsHistBin30_InChebyStats,Image\x7CHistogram_32BinsHistBin30_FourierChebyStats,Image\x7CHistogram_32BinsHistBin31_InChebyStats,Image\x7CHistogram_32BinsHistBin31_FourierChebyStats,Image\x7CHistogram_7BinsHistBin5_InFourierCheby,Image\x7CHistogram_7BinsHistBin5_InCheby,Image\x7CHistogram_7BinsHistBin5_InFourierWavelet,Image\x7CHistogram_7BinsHistBin5_InFourier,Image\x7CHistogram_7BinsHistBin5_In,Image\x7CHistogram_7BinsHistBin5_InWavelet,Image\x7CHistogram_7BinsHistBin4_InFourierCheby,Image\x7CHistogram_7BinsHistBin4_InCheby,Image\x7CHistogram_7BinsHistBin4_InFourierWavelet,Image\x7CHistogram_7BinsHistBin4_InFourier,Image\x7CHistogram_7BinsHistBin4_In,Image\x7CHistogram_7BinsHistBin4_InWavelet,Image\x7CHistogram_32BinsHistBin12_InChebyStats,Image\x7CHistogram_32BinsHistBin12_FourierChebyStats,Image\x7CHistogram_7BinsHistBin6_InFourierCheby,Image\x7CHistogram_7BinsHistBin6_InCheby,Image\x7CHistogram_7BinsHistBin6_InFourierWavelet,Image\x7CHistogram_7BinsHistBin6_InFourier,Image\x7CHistogram_7BinsHistBin6_In,Image\x7CHistogram_7BinsHistBin6_InWavelet,Image\x7CHistogram_7BinsHistBin1_InFourierCheby,Image\x7CHistogram_7BinsHistBin1_InCheby,Image\x7CHistogram_7BinsHistBin1_InFourierWavelet,Image\x7CHistogram_7BinsHistBin1_InFourier,Image\x7CHistogram_7BinsHistBin1_In,Image\x7CHistogram_7BinsHistBin1_InWavelet,Image\x7CHistogram_7BinsHistBin0_InFourierCheby,Image\x7CHistogram_7BinsHistBin0_InCheby,Image\x7CHistogram_7BinsHistBin0_InFourierWavelet,Image\x7CHistogram_7BinsHistBin0_InFourier,Image\x7CHistogram_7BinsHistBin0_In,Image\x7CHistogram_7BinsHistBin0_InWavelet,Image\x7CHistogram_7BinsHistBin3_InFourierCheby,Image\x7CHistogram_7BinsHistBin3_InCheby,Image\x7CHistogram_7BinsHistBin3_InFourierWavelet,Image\x7CHistogram_7BinsHistBin3_InFourier,Image\x7CHistogram_7BinsHistBin3_In,Image\x7CHistogram_7BinsHistBin3_InWavelet,Image\x7CHistogram_7BinsHistBin2_InFourierCheby,Image\x7CHistogram_7BinsHistBin2_InCheby,Image\x7CHistogram_7BinsHistBin2_InFourierWavelet,Image\x7CHistogram_7BinsHistBin2_InFourier,Image\x7CHistogram_7BinsHistBin2_In,Image\x7CHistogram_7BinsHistBin2_InWavelet,Image\x7CHistogram_32BinsHistBin18_InChebyStats,Image\x7CHistogram_32BinsHistBin18_FourierChebyStats,Image\x7CHistogram_32BinsHistBin14_InChebyStats,Image\x7CHistogram_32BinsHistBin14_FourierChebyStats,Image\x7CHistogram_32BinsHistBin15_InChebyStats,Image\x7CHistogram_32BinsHistBin15_FourierChebyStats,Image\x7CHistogram_32BinsHistBin8_InChebyStats,Image\x7CHistogram_32BinsHistBin8_FourierChebyStats,Image\x7CHistogram_32BinsHistBin9_InChebyStats,Image\x7CHistogram_32BinsHistBin9_FourierChebyStats,Image\x7CHistogram_32BinsHistBin16_InChebyStats,Image\x7CHistogram_32BinsHistBin16_FourierChebyStats,Image\x7CHistogram_32BinsHistBin2_InChebyStats,Image\x7CHistogram_32BinsHistBin2_FourierChebyStats,Image\x7CHistogram_32BinsHistBin3_InChebyStats,Image\x7CHistogram_32BinsHistBin3_FourierChebyStats,Image\x7CHistogram_32BinsHistBin0_InChebyStats,Image\x7CHistogram_32BinsHistBin0_FourierChebyStats,Image\x7CHistogram_32BinsHistBin1_InChebyStats,Image\x7CHistogram_32BinsHistBin1_FourierChebyStats,Image\x7CHistogram_32BinsHistBin6_InChebyStats,Image\x7CHistogram_32BinsHistBin6_FourierChebyStats,Image\x7CHistogram_32BinsHistBin7_InChebyStats,Image\x7CHistogram_32BinsHistBin7_FourierChebyStats,Image\x7CHistogram_32BinsHistBin4_InChebyStats,Image\x7CHistogram_32BinsHistBin4_FourierChebyStats,Image\x7CHistogram_32BinsHistBin5_InChebyStats,Image\x7CHistogram_32BinsHistBin5_FourierChebyStats,Image\x7CHistogram_9BinsHistBin7_InFourierCheby,Image\x7CHistogram_9BinsHistBin7_InCheby,Image\x7CHistogram_9BinsHistBin7_InFourierWavelet,Image\x7CHistogram_9BinsHistBin7_InFourier,Image\x7CHistogram_9BinsHistBin7_In,Image\x7CHistogram_9BinsHistBin7_InWavelet,Image\x7CHistogram_9BinsHistBin6_InFourierCheby,Image\x7CHistogram_9BinsHistBin6_InCheby,Image\x7CHistogram_9BinsHistBin6_InFourierWavelet,Image\x7CHistogram_9BinsHistBin6_InFourier,Image\x7CHistogram_9BinsHistBin6_In,Image\x7CHistogram_9BinsHistBin6_InWavelet,Image\x7CHistogram_9BinsHistBin5_InFourierCheby,Image\x7CHistogram_9BinsHistBin5_InCheby,Image\x7CHistogram_9BinsHistBin5_InFourierWavelet,Image\x7CHistogram_9BinsHistBin5_InFourier,Image\x7CHistogram_9BinsHistBin5_In,Image\x7CHistogram_9BinsHistBin5_InWavelet,Image\x7CHistogram_9BinsHistBin4_InFourierCheby,Image\x7CHistogram_9BinsHistBin4_InCheby,Image\x7CHistogram_9BinsHistBin4_InFourierWavelet,Image\x7CHistogram_9BinsHistBin4_InFourier,Image\x7CHistogram_9BinsHistBin4_In,Image\x7CHistogram_9BinsHistBin4_InWavelet,Image\x7CHistogram_9BinsHistBin3_InFourierCheby,Image\x7CHistogram_9BinsHistBin3_InCheby,Image\x7CHistogram_9BinsHistBin3_InFourierWavelet,Image\x7CHistogram_9BinsHistBin3_InFourier,Image\x7CHistogram_9BinsHistBin3_In,Image\x7CHistogram_9BinsHistBin3_InWavelet,Image\x7CHistogram_9BinsHistBin2_InFourierCheby,Image\x7CHistogram_9BinsHistBin2_InCheby,Image\x7CHistogram_9BinsHistBin2_InFourierWavelet,Image\x7CHistogram_9BinsHistBin2_InFourier,Image\x7CHistogram_9BinsHistBin2_In,Image\x7CHistogram_9BinsHistBin2_InWavelet,Image\x7CHistogram_9BinsHistBin1_InFourierCheby,Image\x7CHistogram_9BinsHistBin1_InCheby,Image\x7CHistogram_9BinsHistBin1_InFourierWavelet,Image\x7CHistogram_9BinsHistBin1_InFourier,Image\x7CHistogram_9BinsHistBin1_In,Image\x7CHistogram_9BinsHistBin1_InWavelet,Image\x7CHistogram_9BinsHistBin0_InFourierCheby,Image\x7CHistogram_9BinsHistBin0_InCheby,Image\x7CHistogram_9BinsHistBin0_InFourierWavelet,Image\x7CHistogram_9BinsHistBin0_InFourier,Image\x7CHistogram_9BinsHistBin0_In,Image\x7CHistogram_9BinsHistBin0_InWavelet,Image\x7CHistogram_9BinsHistBin8_InFourierCheby,Image\x7CHistogram_9BinsHistBin8_InCheby,Image\x7CHistogram_9BinsHistBin8_InFourierWavelet,Image\x7CHistogram_9BinsHistBin8_InFourier,Image\x7CHistogram_9BinsHistBin8_In,Image\x7CHistogram_9BinsHistBin8_InWavelet,Image\x7CHistogram_32BinsHistBin17_InChebyStats,Image\x7CHistogram_32BinsHistBin17_FourierChebyStats,Image\x7CHistogram_32BinsHistBin29_InChebyStats,Image\x7CHistogram_32BinsHistBin29_FourierChebyStats,Image\x7CHistogram_32BinsHistBin28_InChebyStats,Image\x7CHistogram_32BinsHistBin28_FourierChebyStats,Image\x7CHistogram_32BinsHistBin24_InChebyStats,Image\x7CHistogram_32BinsHistBin24_FourierChebyStats,Image\x7CHistogram_32BinsHistBin25_InChebyStats,Image\x7CHistogram_32BinsHistBin25_FourierChebyStats,Image\x7CHistogram_32BinsHistBin13_InChebyStats,Image\x7CHistogram_32BinsHistBin13_FourierChebyStats,Image\x7CHistogram_32BinsHistBin27_InChebyStats,Image\x7CHistogram_32BinsHistBin27_FourierChebyStats,Image\x7CHistogram_32BinsHistBin26_InChebyStats,Image\x7CHistogram_32BinsHistBin26_FourierChebyStats,Image\x7CHistogram_32BinsHistBin21_InChebyStats,Image\x7CHistogram_32BinsHistBin21_FourierChebyStats,Image\x7CHistogram_32BinsHistBin20_InChebyStats,Image\x7CHistogram_32BinsHistBin20_FourierChebyStats,Image\x7CHistogram_32BinsHistBin23_InChebyStats,Image\x7CHistogram_32BinsHistBin23_FourierChebyStats,Image\x7CHistogram_32BinsHistBin22_InChebyStats,Image\x7CHistogram_32BinsHistBin22_FourierChebyStats,Image\x7CHistogram_5BinsHistBin3_InFourierCheby,Image\x7CHistogram_5BinsHistBin3_InCheby,Image\x7CHistogram_5BinsHistBin3_InFourierWavelet,Image\x7CHistogram_5BinsHistBin3_InFourier,Image\x7CHistogram_5BinsHistBin3_In,Image\x7CHistogram_5BinsHistBin3_InWavelet,Image\x7CHistogram_5BinsHistBin2_InFourierCheby,Image\x7CHistogram_5BinsHistBin2_InCheby,Image\x7CHistogram_5BinsHistBin2_InFourierWavelet,Image\x7CHistogram_5BinsHistBin2_InFourier,Image\x7CHistogram_5BinsHistBin2_In,Image\x7CHistogram_5BinsHistBin2_InWavelet,Image\x7CHistogram_5BinsHistBin1_InFourierCheby,Image\x7CHistogram_5BinsHistBin1_InCheby,Image\x7CHistogram_5BinsHistBin1_InFourierWavelet,Image\x7CHistogram_5BinsHistBin1_InFourier,Image\x7CHistogram_5BinsHistBin1_In,Image\x7CHistogram_5BinsHistBin1_InWavelet,Image\x7CHistogram_5BinsHistBin0_InFourierCheby,Image\x7CHistogram_5BinsHistBin0_InCheby,Image\x7CHistogram_5BinsHistBin0_InFourierWavelet,Image\x7CHistogram_5BinsHistBin0_InFourier,Image\x7CHistogram_5BinsHistBin0_In,Image\x7CHistogram_5BinsHistBin0_InWavelet,Image\x7CHistogram_5BinsHistBin4_InFourierCheby,Image\x7CHistogram_5BinsHistBin4_InCheby,Image\x7CHistogram_5BinsHistBin4_InFourierWavelet,Image\x7CHistogram_5BinsHistBin4_InFourier,Image\x7CHistogram_5BinsHistBin4_In,Image\x7CHistogram_5BinsHistBin4_InWavelet,Image\x7CMoments_Standard Deviation_InFourierCheby,Image\x7CMoments_Standard Deviation_InCheby,Image\x7CMoments_Standard Deviation_InFourierWavelet,Image\x7CMoments_Standard Deviation_InFourier,Image\x7CMoments_Standard Deviation_In,Image\x7CMoments_Standard Deviation_InWavelet,Image\x7CMoments_Kurtosis_InFourierCheby,Image\x7CMoments_Kurtosis_InCheby,Image\x7CMoments_Kurtosis_InFourierWavelet,Image\x7CMoments_Kurtosis_InFourier,Image\x7CMoments_Kurtosis_In,Image\x7CMoments_Kurtosis_InWavelet,Image\x7CMoments_Skewness_InFourierCheby,Image\x7CMoments_Skewness_InCheby,Image\x7CMoments_Skewness_InFourierWavelet,Image\x7CMoments_Skewness_InFourier,Image\x7CMoments_Skewness_In,Image\x7CMoments_Skewness_InWavelet,Image\x7CMoments_Mean_InFourierCheby,Image\x7CMoments_Mean_InCheby,Image\x7CMoments_Mean_InFourierWavelet,Image\x7CMoments_Mean_InFourier,Image\x7CMoments_Mean_In,Image\x7CMoments_Mean_InWavelet,Image\x7CMetadata_Key
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name:CHARM-like_training_data.csv
    Use the object name for the file name?:No

ExportToSpreadsheet:[module_num:19|svn_version:\'10880\'|variable_revision_number:7|show_window:False|notes:\x5B\x5D]
    Select or enter the column delimiter:Comma (",")
    Prepend the output file name to the data file names?:Yes
    Add image metadata columns to your object data file?:No
    Limit output to a size that is allowed in Excel?:No
    Select the columns of measurements to export?:Yes
    Calculate the per-image mean values for object measurements?:No
    Calculate the per-image median values for object measurements?:No
    Calculate the per-image standard deviation values for object measurements?:No
    Output file location:Default Output Folder\x7CNone
    Create a GenePattern GCT file?:No
    Select source of sample row name:Metadata
    Select the image to use as the identifier:None
    Select the metadata to use as the identifier:None
    Export all measurements, using default file names?:No
    Press button to select measurements to export:Image\x7CMetadata_HoldOut,Image\x7CMetadata_Class,Image\x7CMetadata_Key
    Data to export:Image
    Combine these object measurements with those of the previous object?:No
    File name:CHARM-like_training_labels.csv
    Use the object name for the file name?:No
