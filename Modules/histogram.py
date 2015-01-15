'''<b>MeasurementTemplate</b> - an example measurement module
<hr>
This is an example of a module that measures a property of an image both
for the image as a whole and for every object in the image. It demonstrates
how to load an image, how to load an object and how to record a measurement.

The text you see here will be displayed as the help for your module. You
can use HTML markup here and in the settings text; the Python HTML control
does not fully support the HTML specification, so you may have to experiment
to get it to display correctly.
'''
#################################
#
# Imports from useful Python libraries
#
#################################

import numpy as np
import scipy.ndimage as scind

#################################
#
# Imports from CellProfiler
#
# The package aliases are the standard ones we use
# throughout the code.
#
##################################

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps

##################################
#
# Constants
#
# I put constants that are used more than once here.
#
###################################
BINS = (3, 5, 7, 9)

'''This is the measurement template category'''
C_MEASUREMENT_TEMPLATE = "HIST"

def get_histogram(pixels, b):
    bins, junk = np.histogram(pixels, bins=b)
    bins=np.array(bins, dtype=float)
    bins=bins/np.max(bins)
    return bins

###################################
#
# The module class
#
# Your module should "inherit" from cellprofiler.cpmodule.CPModule.
# This means that your module will use the methods from CPModule unless
# you re-implement them. You can let CPModule do most of the work and
# implement only what you need.
#
###################################

class Histogram(cpm.CPModule):
    ###############################################
    #
    # The module starts by declaring the name that's used for display,
    # the category under which it is stored and the variable revision
    # number which can be used to provide backwards compatibility if
    # you add user-interface functionality later.
    #
    ###############################################
    module_name = "Histogram"
    category = "Measurement"
    variable_revision_number = 1
    
    ###############################################
    #
    # create_settings is where you declare the user interface elements
    # (the "settings") which the user will use to customize your module.
    #
    # You can look at other modules and in cellprofiler.settings for
    # settings you can use.
    #
    ################################################
    
    def create_settings(self):
        #
        # The ImageNameSubscriber "subscribes" to all ImageNameProviders in 
        # prior modules. Modules before yours will put images into CellProfiler.
        # The ImageSubscriber gives your user a list of these images
        # which can then be used as inputs in your module.
        #
        self.input_image_name = cps.ImageNameSubscriber(
            # The text to the left of the edit box
            "Input image name:",
            # HTML help that gets displayed when the user presses the
            # help button to the right of the edit box
            doc = """This is the image that the module operates on. You can
            choose any image that is made available by a prior module.
            <br>
            <b>ImageTemplate</b> will do something to this image.
            """)
        
    #
    # The "settings" method tells CellProfiler about the settings you
    # have in your module. CellProfiler uses the list for saving
    # and restoring values for your module when it saves or loads a
    # pipeline file.
    #
    # This module does not have a "visible_settings" method. CellProfiler
    # will use "settings" to make the list of user-interface elements
    # that let the user configure the module. See imagetemplate.py for
    # a template for visible_settings that you can cut and paste here.
    #
    def settings(self):
        return [self.input_image_name]
    
    #
    # CellProfiler calls "run" on each image set in your pipeline.
    # This is where you do the real work.
    #
    def run(self, workspace):
        #
        # Get the measurements object - we put the measurements we
        # make in here
        #
        meas = workspace.measurements
        assert isinstance(meas, cpmeas.Measurements)
        #
        # We record some statistics which we will display later.
        # We format them so that Matplotlib can display them in a table.
        # The first row is a header that tells what the fields are.
        #
        statistics = [ [ "Feature", "Counts"] ]
        #
        # Put the statistics in the workspace display data so we
        # can get at them when we display
        #
        workspace.display_data.statistics = statistics
        #
        # Get the input image and object. You need to get the .value
        # because otherwise you'll get the setting object instead of
        # the string name.
        #
        input_image_name = self.input_image_name.value
        ################################################################
        #
        # GETTING AN IMAGE FROM THE IMAGE SET
        #
        # Get the image set. The image set has all of the images in it.
        # The assert statement makes sure that it really is an image set,
        # but, more importantly, it lets my editor do context-sensitive
        # completion for the image set.
        #
        image_set = workspace.image_set
        assert isinstance(image_set, cpi.ImageSet)
        #
        # Get the input image object. We want a grayscale image here.
        # The image set will convert a color image to a grayscale one
        # and warn the user.
        #
        input_image = image_set.get_image(input_image_name,
                                          must_be_grayscale = True)
        #
        # Get the pixels - these are a 2-d Numpy array.
        #
        pixels = input_image.pixel_data
        
        #
        # The module computes a measurement based on the image intensity
        # inside an object times a Zernike polynomial inscribed in the
        # minimum enclosing circle around the object. The details are
        # in the "measure_zernike" function. We call into the function with
        # an N and M which describe the polynomial.
        #
        for b in BINS:
            # Compute the histogram returned in an array
            z = get_histogram(pixels, b)
            # Get the name of the measurement feature for this histogram
            for i in range(0, len(z)):
                meas.add_measurement(cpmeas.IMAGE, self.get_measurement_name(b, i), z[i])
            #
            # Record the statistics. 
            #
            for i in range(0, len(z)):
                statistics.append( [ self.get_measurement_name(b, i), z[i] ] )
    ################################
    # 
    # DISPLAY
    #
    # We define is_interactive to be False to tell CellProfiler
    # that it should execute "run" in a background thread and then
    # execute "display" in a foreground thread.
    #
    def is_interactive(self):
        return False
    
    def display(self, workspace):
        statistics = workspace.display_data.statistics
        figure = workspace.create_or_find_figure(subplots=(1,1,))
        figure.subplot_table(0,0, statistics, ratio = (0.5, 0.5))
    
    ################################
    #
    # 
    # 

    
    #######################################
    #
    # Here, we go about naming the measurements.
    #
    # Measurement names have parts to them, traditionally separated
    # by underbars. There's always a category and a feature name
    # and sometimes there are modifiers such as the image that
    # was measured or the scale at which it was measured.
    #
    # We have functions that build the names so that we can
    # use the same functions in different places.
    #
    def get_feature_name(self, b, i):
        '''Return a measurement feature name'''
        #
        # Something nice and simple for a name... IntensityN4M2 for instance
        #
        return "%dBinsHistBin%d" % (b, i)
    
    def get_measurement_name(self, b, i):
        '''Return the whole measurement name'''
        input_image_name = self.input_image_name.value
        return '_'.join([C_MEASUREMENT_TEMPLATE, 
                         self.get_feature_name(b, i),
                         input_image_name])
    #
    # We have to tell CellProfiler about the measurements we produce.
    # There are two parts: one that is for database-type modules and one
    # that is for the UI. The first part gives a comprehensive list
    # of measurement columns produced. The second is more informal and
    # tells CellProfiler how to categorize its measurements.
    #
    #
    # get_measurement_columns gets the measurements for use in the database
    # or in a spreadsheet. Some modules need the pipeline because they
    # might make measurements of measurements and need those names.
    #
    def get_measurement_columns(self, pipeline):
        #
        # We use a list comprehension here.
        # See http://docs.python.org/tutorial/datastructures.html#list-comprehensions
        # for how this works.
        #
        # The first thing in the list is the object being measured. If it's
        # the whole image, use cpmeas.IMAGE as the name.
        #
        # The second thing is the measurement name.
        #
        # The third thing is the column type. See the COLTYPE constants
        # in measurements.py for what you can use
        #
        return [ (cpmeas.IMAGE,
                  self.get_measurement_name(b, i),
                  cpmeas.COLTYPE_FLOAT)
                 for b in BINS for i in range(0, b)]
    
    #
    # get_categories returns a list of the measurement categories produced
    # by this module. It takes an object name - only return categories
    # if the name matches.
    #
    def get_categories(self, pipeline, image_name):
        if image_name == self.input_image_name:
            return [ C_MEASUREMENT_TEMPLATE ]
        else:
            # Don't forget to return SOMETHING! I do this all the time
            # and CP mysteriously bombs when you use ImageMath
            return []

    #
    # Return the feature names if the object_name and category match
    #
    def get_measurements(self, pipeline, image_name, category):
        if (image_name == self.input_image_name and
            category == C_MEASUREMENT_TEMPLATE):
            #
            # Use another list comprehension. See docs in get_measurement_columns.
            return [ self.get_feature_name(b, i)
                     for b in BINS for i in range(0, b)]
        else:
            return []
        
    #
    # This module makes per-image measurements. That means we need
    # get_measurement_images to distinguish measurements made on two
    # different images by this module
    #
    def get_measurement_images(self, pipeline, image_name, category, measurement):
        #
        # This might seem wasteful, but UI code can be slow. Just see
        # if the measurement is in the list returned by get_measurements
        #
        if measurement in self.get_measurements(
            pipeline, image_name, category):
            return [ self.input_image_name.value ]
        else: 
            return []
