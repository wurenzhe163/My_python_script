# -*- coding: utf-8 -*-

import arcpy
import gdal,os
search_files = lambda path, endwith: [os.path.join(path, f) for f in os.listdir(path) if f.endswith(endwith)]

class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Toolbox"
        self.alias = ""

        # List of tool classes associated with this toolbox
        self.tools = [Tool]


class Tool(object):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "XML批量生成"
        self.description = "根据图像和标注生成对应的xml以及图像切片"
        self.canRunInBackground = False

    def getParameterInfo(self):
        """Define parameter definitions"""
        # First parameter
        param0 = arcpy.Parameter(
            displayName="Input ImageSpace",
            name="in_imagespace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")

        param1 = arcpy.Parameter(
            displayName="Input LabelSpace",
            name="in_Labelpace",
            datatype="DEWorkspace",
            parameterType="Required",
            direction="Input")
      

        param2 = arcpy.Parameter(
            displayName="Output Dir",
            name="out_dir",
            datatype="DEWorkspace",
            parameterType="Derived",
            direction="Output")

        # # Set the filter to accept only local (personal or file) geodatabases
        # param0.filter.list = ["Local Database"]
        # # param2.parameterDependencies = [param0.name]
        # # param2.schema.clone = True
        param2.parameterDependencies = [param0.name]
        param2.schema.clone = True  
        params = [param0, param1, param2]
        return params

    def isLicensed(self):
        """Set whether tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter.  This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""
        image_path = search_files(parameters[0].valueAsText,'.tif')
        label_path = search_files(parameters[1].valueAsText,'.shp')
        
        output_path = parameters[2].valueAsText
        
        for each_image,each_lable in zip(image_path,label_path):
            arcpy.sa.ExportTrainingDataForDeepLearning(each_image,
                                                      output_path,
        os.path.join(os.path.dirname(each_lable) ,os.path.basename(each_lable).split('.')[0]),
            "TIFF", 512, 512, 256, 256,
            "ONLY_TILES_WITH_FEATURES", "Classified_Tiles",
             0, "Category_C", 0, None,
              0, "IMAGE_SPACE", 
              "PROCESS_AS_MOSAICKED_IMAGE",
               "NO_BLACKEN", "FIXED_SIZE")
                                                      
        return True

