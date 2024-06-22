using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using AP_SelfHealing.MachineLearning.DataModels;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Text;

namespace AP_SelfHealing
{
    public static class ElementPrediction1
    {
        public static void FindAttribute()
        {
            // Initialize ML.NET context
            var mlContext = new MLContext();
            // Load data
            var df = mlContext.Data.LoadFromTextFile<Model3>("C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\file.csv", separatorChar: ',', hasHeader: true);
            Console.WriteLine(df.ToString());
            var test = mlContext.Data.LoadFromTextFile<Model3>("C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Test.csv", separatorChar: ',', hasHeader: true);
            Console.WriteLine(test.ToString());
            // Fill the NaN values with 'None'
            //df = mlContext.Data.FilterRowsByMissingValues( (df, nameof(Model.Element), 0, 0, true);
            // Define data preprocessing pipeline
            //var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(nameof(Model.Element)) 
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Model.Feature1))) 
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Model.Feature2))) 
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Model.Feature3))) 
            //    .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(Model.Feature4)));

            //context = new MLContext();
            //// Reading the required files

            ////LoadFromTextFile is lazy load, so no actual data load here only schema validation
            //IDataView trainData = context.Data.LoadFromTextFile<Model>(
            //    "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\file.csv",
            //    hasHeader: true);
            //Console.WriteLine(trainData);

            ////Split data into train and test
            ////DataOperationsCatalog.TrainTestData _dataSplit = context.Data.TrainTestSplit(data, testFraction: 0.1);
            ////var trainData = _dataSplit.TrainSet;
            ////Console.WriteLine(trainData);
            ////The features are the input you want to use to make a prediction, the label is the data you want to predict
            ////Filter the data - remove unnesscary column
            ////https://learn.microsoft.com/en-us/dotnet/machine-learning/how-to-guides/prepare-data-ml-net
            ////Not performing any filter

            ////Fill the null value with None           
            //// Replace missing values
            //var columns = trainData.Schema.Select(column => column.Name).ToArray();
            ////.Where(colName => colName != "Element").ToArray();
            //var nullTransform = context.Transforms.Concatenate("Features", columns);
            ////    Run and check concatenation output
            ////Console.WriteLine("After concatenation");
            ////Console.WriteLine(nullTransform.Preview(trainData));
            //var transformedData = nullTransform.Fit(trainData).Transform(trainData);

            //IEnumerable<TransformedData> featuresColumn = context.Data.CreateEnumerable<TransformedData>(
            //       transformedData, reuseRowObject: false);
            ////// And we can write out a few rows
            //Console.WriteLine($"Features column obtained post-transformation.");
            //foreach (var featureRow in featuresColumn)
            //    Console.WriteLine(string.Join(" ", featureRow.Features));

            ////Missing Indicator but this is only required for numberic data

            ////var nullTransform = context.Transforms.IndicateMissingValues(new[]
            ////{
            ////    new InputOutputColumnPair("MissingIndicator1", "Line"),
            ////    new InputOutputColumnPair("MissingIndicator2", "LinePosition"),
            ////    new InputOutputColumnPair("MissingIndicator3", "ControlId"),
            ////    new InputOutputColumnPair("MissingIndicator4", "Name"),
            ////    new InputOutputColumnPair("MissingIndicator5", "CCSClass"),
            ////    new InputOutputColumnPair("MissingIndicator6", "Value"),
            ////    new InputOutputColumnPair("MissingIndicator7", "Label"),
            ////    new InputOutputColumnPair("MissingIndicator8", "Role"),
            ////    new InputOutputColumnPair("MissingIndicator9", "Type"),
            ////    new InputOutputColumnPair("MissingIndicator10", "TabIndex"),
            ////    new InputOutputColumnPair("MissingIndicator11", "Placeholder"),
            ////    new InputOutputColumnPair("MissingIndicator12", "Title"),
            ////    new InputOutputColumnPair("MissingIndicator13", "Href"),
            ////});
            ////var tansformer = nullTransform.Fit(trainData);
            ////var transformedData = tansformer.Transform(trainData);

            ////var rowEnumerable = context.Data.CreateEnumerable<
            ////        SampleDataTransformed>(transformedData, reuseRowObject: false);

            ////foreach (var row in rowEnumerable)
            ////    Console.WriteLine("Line: [" + string.Join(", ", row
            ////        .Line) + "]\t MissingIndicator1: [" + string.Join(", ",
            ////        row.MissingIndicator1) + "]\t LinePosition: [" + string.Join(", ",
            ////        row.LinePosition) + "]\t MissingIndicator2: [" + string.Join(", ",
            ////        row.MissingIndicator2) + "]\t ControlId: [" + string.Join(", ",
            ////        row.ControlId) + "]\t MissingIndicator3: [" + string.Join(", ",
            ////        row.MissingIndicator3) + "]\t Name: [" + string.Join(", ",
            ////        row.Name ) + "]\t MissingIndicator4: [" + string.Join(", ",
            ////        row.MissingIndicator4) + "]");



            ////ML.NET machine learning algorithms expect input or features to be in a single numerical vector.
            ////Convert text data to numberical features

            ////// A pipeline for converting text into numeric features.
            //// The following call to 'FeaturizeText' instantiates
            //// 'TextFeaturizingEstimator' with given parameters. The length of the
            //// output feature vector depends on these settings.
            //var options = new TextFeaturizingEstimator.Options()
            //{
            //    // Also output tokenized words
            //    OutputTokensColumnName = "OutputTokens",
            //    CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            //    // Use ML.NET's built-in stop word remover
            //    StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options()
            //    {
            //        Language = TextFeaturizingEstimator.Language.English
            //    },

            //    WordFeatureExtractor = new WordBagEstimator.Options()
            //    {
            //        NgramLength
            //        = 2,
            //        UseAllLengths = true
            //    },

            //    CharFeatureExtractor = new WordBagEstimator.Options()
            //    {
            //        NgramLength
            //        = 3,
            //        UseAllLengths = false
            //    },
            //};

            //var textEsimator = context.Transforms.Text.FeaturizeText("FeaturesNum", options, "Features");


            //// Fit to data.
            //var testSetTransform = textEsimator.Fit(transformedData).Transform(transformedData);

            //// Define trainer options.
            //var optionsFF = new FastForestRegressionTrainer.Options
            //{
            //    //LabelColumnName = nameof(TransformedTextData.Element),
            //    FeatureColumnName = nameof(TransformedTextData.FeaturesNum),
            //    // Only use 80% of features to reduce over-fitting.
            //    FeatureFraction = 0.8,
            //    // Create a simpler model by penalizing usage of new features.
            //    FeatureFirstUsePenalty = 0.1,
            //    // Reduce the number of trees to 50.
            //    NumberOfTrees = 50
            //};

            ////Define the trainer
            //var pipeline = context.Regression.Trainers.FastForest(optionsFF);

            ////Train the model
            //var model = pipeline.Fit(testSetTransform);

            //IDataView testData = context.Data.LoadFromTextFile<Model>(
            //    "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\test.csv",
            //    hasHeader: true);

            ////Run the model on the test data set.
            //var transformedTestData = model.Transform(testData);

            //// Convert IDataView object to a list.
            //var predictions = context.Data.CreateEnumerable<Prediction>(
            //    transformedTestData, reuseRowObject: false).ToList();
            //Console.WriteLine($"Label: {predictions[0].Label:F3}, Prediction: {predictions[0].Score:F3}");
            //// Create the prediction engine to get the features extracted from the
            //// text.
            ////If you this, this is threadless 
            ////var predictionEngine = context.Model.CreatePredictionEngine<TransformedData,
            ////    TransformedTextData>(textTransformer);

            //// Convert the text into numeric features.
            ////int count = 1;
            ////foreach (var col in featuresColumn)
            ////{
            ////    Console.WriteLine(count +" row");uvgbgv
            ////    var prediction = predictionEngine.Predict(col);

            ////    // Print the length of the feature vector.
            ////    Console.WriteLine($"Number of FeaturesNum: {prediction.FeaturesNum.Length}");

            ////    // Print feature values and tokens.
            ////    Console.Write("FeaturesNum: ");
            ////    for (int i = 0; i < 10; i++)
            ////        Console.Write($"{prediction.FeaturesNum[i]:F4}  ");

            ////    Console.WriteLine("\nTokens: " + string.Join(",", prediction
            ////        .OutputTokens));
            ////    count++;
            ////}
        }
    }
}

 // Train the model var trainingPipeline = dataProcessPipeline.Append(mlContext.MulticlassClassification.Trainers .RandomForest(numTrees: 50) .Append(mlContext.Transforms.Conversion.MapKeyToValue(nameof(Model.Element)))); var trainedModel = trainingPipeline.Fit(df); // Define prediction engine var predictionEngine = mlContext.Model.CreatePredictionEngine<Model, ModelPrediction>(trainedModel); // Define function to predict elements (List<(string, float[])> scores, List<string> elementNames, IDataView testData) PredictElements() { var numRecords = test.GetColumn<Model>(nameof(Model.Element)).Length; var probabilities = new List<(string, float[])>(); var elementNames = new List<string>(); if (numRecords == 1) { var prediction = predictionEngine.Predict(test); var elementName = mlContext.Data.GetClasses<Model>(test.Schema, nameof(Model.Element)) [prediction.Element]; elementNames.Add($"Hence, the name of our predicted element is {elementName}"); probabilities.Add((elementName, prediction.Score)); } else if (numRecords > 1) { var predictions = trainedModel.Transform(test); var predictionData = mlContext.Data.CreateEnumerable<ModelPrediction>(predictions, false); foreach (var prediction in predictionData) { var elementName = mlContext.Data.GetClasses<Model>(test.Schema, nameof(Model.Element)) [prediction.Element]; elementNames.Add($"Hence, the name of our predicted element is {elementName}"); probabilities.Add((elementName, prediction.Score)); } } return (probabilities, elementNames, test); } // Calling the PredictElements method to return scores, elementNames, and testData var (scores, elementNames, testData) = PredictElements(); // Output scores and elementNames Console.WriteLine("Scores:"); foreach (var (element, score) in scores) { Console.WriteLine($"{element}: {string.Join(", ", score)}"); } Console.WriteLine("Element Names:"); foreach (var name in elementNames) { Console.WriteLine(name); } } }
