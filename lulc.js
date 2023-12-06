var pokhara = ee.FeatureCollection("users/prashantthapaliya8989/LST/pokhara"),
    imageVisParam = {"opacity":1,"bands":["B4","B3","B2"],"min":-48.76734084518648,"max":3000,"gamma":1},
    trueColor432Vis = {"min":0,"max":3000},
    builtup = ee.FeatureCollection("users/prashantthapaliya8989/LST/builtup"),
    cultivatedLand = ee.FeatureCollection("users/prashantthapaliya8989/LST/cultivatedLand"),
    forest = ee.FeatureCollection("users/prashantthapaliya8989/LST/forest"),
    waterBody = ee.FeatureCollection("users/prashantthapaliya8989/LST/waterBody");

//Image Collection of Year 2020
var landsat=ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA")
    .filterBounds(pokhara)
    .filter(ee.Filter.lt('CLOUD_COVER', 5))
    .filterDate('2020-03-01','2020-06-30')
    .first();
var landsat_2020=landsat.clip(pokhara);
print ("Landsat 2020 Image", landsat_2020);

Map.setCenter(83.9812, 28.2203, 10);  
Map.addLayer(landsat_2020, {bands: ['B5', 'B4', 'B3']}, "Landsat 8");
Map.addLayer(pokhara,{},"Pokhara Boundary",false);
//merge the feature collection
var landCover = forest.merge(cultivatedLand).merge(builtup).merge(waterBody)
Map.addLayer(landCover,{},"Landcover",false);
print ("Landcover",landCover);

//select Bands from mosaic Image for training
var bands=['B2','B3','B4','B5', 'B6','B7'];
var classProperty = 'landCover';

var training = landsat_2020.select(bands).sampleRegions({
  collection:landCover,
  properties:['landCover'],
  scale:10
});

print("Training", training);

//Train the classifier
var classifier = ee.Classifier.smileCart().train({
  features: training,
  classProperty: classProperty,
  inputProperties:bands
});

//Classify the input imagery
var classified_2020 = landsat_2020.select(bands).classify(classifier);
print(classified_2020);

var palette=['#0a7505','#ffe2a7', 'ff0000', '468DFF'];
Map.addLayer(classified_2020,{min:1,max:4,palette:palette},"classified_2020");


var withRandom= training.randomColumn('random');
print("Total counts ",withRandom.getInfo()['features'].length);

// we want to reverse some to the data for testing, to avoid overfitting the model.
var split = 0.7; //Roughly 70% training, 30% testing
var trainingPartition=withRandom.filter(ee.Filter.lt('random',split));
var testingPartition=withRandom.filter(ee.Filter.gte('random',split));

print("Total trainingPartition ",trainingPartition.getInfo()['features'].length);
print("Total testingPartition ",testingPartition.getInfo()['features'].length);

// Trained with 70% of our data. 
var trainedClassifier= ee.Classifier.smileCart().train({
  features: trainingPartition,
  classProperty: classProperty,
  inputProperties:bands
});

//Classify the trained imagery
var trained_2020 = landsat_2020.select(bands).classify(trainedClassifier);

//print the confusion Matrix.
var trainAccuracy=trainedClassifier.confusionMatrix();
// print('Confusion Matrix ',confusionMatrix_2019);
print('Resubstitution error matrix: ', trainAccuracy);
print('Training overall accuracy: ', trainAccuracy.accuracy());

// classify the test FeatureCollection.
var validated = testingPartition.classify(trainedClassifier);
var testAccuracy = validated.errorMatrix(classProperty,'classification');
// print('Validation error matrix: ', testAccuracy);
print('Validation overall accuracy: ', testAccuracy.accuracy());


//Export the LULC Map
Export.image.toDrive({
  image: classified_2020,
  description: 'classification_Pokhara',
  scale: 30,
  region: pokhara,
  folder : 'LULC Map',
  maxPixels: 3784216672400,
  crs : 'EPSG:4326'
});
