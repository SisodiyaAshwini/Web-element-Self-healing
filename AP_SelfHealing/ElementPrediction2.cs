using Microsoft.ML;
using AP_SelfHealing.MachineLearning.DataModels;
using System.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Text;
using Microsoft.ML.Data;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AP_SelfHealing
{
    public class getlabel : Model
    {
        public float LabelToPred { get; set; }
    }
    public class TransformedData : getlabel
    {
        [VectorType]
        public string[] Features { get; set; }
    }

    public class TransformedTextData : TransformedData
    {
        [VectorType]
        public float[] FeaturesNum { get; set; }
        public string[] OutputTokens { get; set; }
        public float ElementNum { get; set; }
    }

    public class Prediction
    {
        // Original label.
        public float Label { get; set; }
        // Predicted score from the trainer.
        public float Score { get; set; }
    }
    public static class ElementPrediction2
    {
        public static MLContext context;
        
        public static void LoadAndPrepareData()
        {
            context = new MLContext();
            // Reading the required files

            //LoadFromTextFile is lazy load, so no actual data load here only schema validation
            IDataView trainData = context.Data.LoadFromTextFile<Model>(
                "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\file.csv",
                hasHeader: true);
            Console.WriteLine(trainData);

            //Convert label(i.e. Element) from text to float
            //Get distinct labels
            var distinctLabels = context.Data.CreateEnumerable<Model>(trainData, reuseRowObject: false)
                .Select(x => x.Element).Distinct().ToList();

            //Create a dictionary to map text label to float
            var labelToFloatmap = distinctLabels.Select((Element, index) => new { Element, index })
                .ToDictionary(x => x.Element, x => (float)x.index + 1);

            //Define the custom mapping to convert text label(i.i element) to float
            Action<Model, getlabel> customMapping = (input, output) =>
            {
                output.LabelToPred = labelToFloatmap.ContainsKey(input.Element) ? labelToFloatmap[input.Element] : float.NaN;
            };


            //Add a property "LabeltoPred" to TransformedTextData class
            var pipeline1 = context.Transforms.CustomMapping(customMapping, "CustomMapping");
            var transformer = pipeline1.Fit(trainData).Transform(trainData);
            var convertedData = context.Data.CreateEnumerable<getlabel>(
                    transformer, true);

            //// Printing the results.
            //Console.WriteLine("Element");
            //foreach (var item in convertedData)
            //    Console.WriteLine(item.Element +" - "+item.LabelToPred);

            //Concatenate input columns
            var columns = trainData.Schema.Select(column => column.Name)
            .Where(colName => colName != "Element").ToArray();

            var nullTransform = context.Transforms.Concatenate("Features", columns);

            //var transformedData = nullTransform.Fit(trainData).Transform(trainData);
            var transformedData = nullTransform.Fit(transformer).Transform(transformer);

            IEnumerable<TransformedData> featuresColumn = context.Data.CreateEnumerable<TransformedData>(
                   transformedData, reuseRowObject: false);
            //// And we can write out a few rows
            Console.WriteLine($"Features column obtained post-transformation.");
            foreach (var featureRow in featuresColumn)
                Console.WriteLine(string.Join(" ", featureRow.Features));

            //ML.NET machine learning algorithms expect input or features to be in a single numerical vector.
            //Convert text data to numberical features

            //// A pipeline for converting text into numeric features.
            // The following call to 'FeaturizeText' instantiates
            // 'TextFeaturizingEstimator' with given parameters. The length of the
            // output feature vector depends on these settings.
            var options = new TextFeaturizingEstimator.Options()
            {
                // Also output tokenized words
                OutputTokensColumnName = "OutputTokens",
                CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                // Use ML.NET's built-in stop word remover
                StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options()
                {
                    Language = TextFeaturizingEstimator.Language.English
                },

                WordFeatureExtractor = new WordBagEstimator.Options()
                {
                    NgramLength
                    = 2,
                    UseAllLengths = true
                },

                CharFeatureExtractor = new WordBagEstimator.Options()
                {
                    NgramLength
                    = 3,
                    UseAllLengths = false
                },
            };

            var textEsimator = context.Transforms.Text.FeaturizeText("FeaturesNum", options, "Features");
                

            // Fit to data.
            var testSetTransform = textEsimator.Fit(transformedData).Transform(transformedData);

            //for label
            var textEsimatorlbl = context.Transforms.Text.FeaturizeText("ElementNum", options, "Element");
            var testSetTransformlbl = textEsimatorlbl.Fit(testSetTransform).Transform(testSetTransform);

            // Define trainer options.
            var optionsFF = new FastForestRegressionTrainer.Options
            {
                LabelColumnName = nameof(TransformedTextData.ElementNum),
                FeatureColumnName = nameof(TransformedTextData.FeaturesNum),
                // Only use 80% of features to reduce over-fitting.
                FeatureFraction = 0.8,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.1,
                // Reduce the number of trees to 50.
                NumberOfTrees = 50
            };

            //Define the trainer
            var pipeline = context.Regression.Trainers.FastForest(optionsFF);

            //Train the model
            var model = pipeline.Fit(testSetTransformlbl);

            IDataView testData = context.Data.LoadFromTextFile<Model>(
                "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\test.csv",
                hasHeader: true);

            //Run the model on the test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = context.Data.CreateEnumerable<ModelPrediction>(
                transformedTestData, reuseRowObject: false).ToList();
            Console.WriteLine($"Label: {predictions[0].Element:F3}, Prediction: {predictions[0].Score:F3}");

        }
    }
}