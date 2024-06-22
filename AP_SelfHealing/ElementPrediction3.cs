﻿using Microsoft.ML;
using AP_SelfHealing.MachineLearning.DataModels;
using System.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms.Text;
using System;

namespace AP_SelfHealing
{
    public static class ElementPrediction3
    {
        public static MLContext context;

        public static void LoadAndPrepareData()
        {
            context = new MLContext();
            // Reading the required files

            //LoadFromTextFile is lazy load, so no actual data load here only schema validation
            IDataView trainData = context.Data.LoadFromTextFile<Model3>(
                "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\file.csv",
                hasHeader: true);

            //Convert label(i.e. Element) from text to float
            //Get distinct labels
            var distinctLabels = context.Data.CreateEnumerable<Model3>(trainData, reuseRowObject: false)
                .Select(x => x.Element).Distinct().ToList();

            //Create a dictionary to map text label to float
            var labelToFloatmap = distinctLabels.Select((Element, index) => new { Element, index })
                .ToDictionary(x => x.Element, x => (float)x.index + 1);

            //Define the custom mapping to convert text label(i.i element) to float
            Action<Model3, TransformedData> customMapping = (input, output) =>
            {
                output.Line = input.Line;
                output.LinePosition = input.LinePosition;
                output.Role = input.Role;
                output.Title = input.Title;
                output.CCSClass = input.CCSClass;
                output.ControlId = input.ControlId;
                output.Name = input.Name;
                output.Href = input.Href;
                output.Value = input.Value;
                output.Role = input.Role;
                output.Type = input.Type;
                output.TabIndex = input.TabIndex;
                output.Placeholder = input.Placeholder;
                output.Href = input.Href;
                output.Label = labelToFloatmap.ContainsKey(input.Element) ? labelToFloatmap[input.Element] : float.NaN;
            };

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

            //Add a property "LabeltoPred" to TransformedTextData class
            //var pipeline = context.Transforms.CustomMapping(customMapping, "CustomMapping")
            //    .Append(context.Transforms.Concatenate("Features", new[] { "Line", "LinePosition", "ControlId", "Name", "CCSClass"
            //    , "Value", "Role", "Type","TabIndex","Placeholder","Title","Href" }))
            //    .Append(context.Transforms.Text.FeaturizeText("FeaturesNum", options, "Features"));
            var pipeline = context.Transforms.CustomMapping(customMapping, "CustomMapping")
                .Append(context.Transforms.Text.FeaturizeText("LineF", options, "Line"))
                .Append(context.Transforms.Text.FeaturizeText("LinePositionF", options, "LinePosition"))
                .Append(context.Transforms.Text.FeaturizeText("ControlIdF", options, "ControlId"))
                .Append(context.Transforms.Text.FeaturizeText("NameF", options, "Name"))
                .Append(context.Transforms.Text.FeaturizeText("CCSClassF", options, "CCSClass"))
                .Append(context.Transforms.Text.FeaturizeText("ValueF", options, "Value"))
                .Append(context.Transforms.Text.FeaturizeText("RoleF", options, "Role"))
                .Append(context.Transforms.Text.FeaturizeText("TypeF", options, "Type"))
                .Append(context.Transforms.Text.FeaturizeText("TabIndexF", options, "TabIndex"))
                .Append(context.Transforms.Text.FeaturizeText("PlaceholderF", options, "Placeholder"))
                .Append(context.Transforms.Text.FeaturizeText("TitleF", options, "Title"))
                .Append(context.Transforms.Text.FeaturizeText("HrefF", options, "Href"))
                .Append(context.Transforms.Concatenate("Features", new[] { "LineF", "LinePositionF", "ControlIdF", "NameF", "CCSClassF"
                , "ValueF", "RoleF", "TypeF","TabIndexF","PlaceholderF","TitleF","HrefF" }));
            var transformer = pipeline.Fit(trainData).Transform(trainData);

            // Define trainer options./'9
            var optionsFF = new FastForestRegressionTrainer.Options
            {
                LabelColumnName = nameof(TransformedDataF.Label),
                FeatureColumnName = nameof(TransformedDataF.Features),
                // Only use 80% of features to reduce over-fitting.
                FeatureFraction = 0.8,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.1,
                // Reduce the number of trees to 50.
                NumberOfTrees = 50
            };

            //Define the trainer
            //Trainer = Algorithm + Task
            //An algorithm is the math that executes to produce a model
            var pipelineFF = context.Regression.Trainers.FastForest(optionsFF);

            //Train the model
            var model = pipelineFF.Fit(transformer);

            //IEnumerable<TransformedDataF> featuresColumn = context.Data.CreateEnumerable<TransformedDataF>(
            //       transformer, reuseRowObject: false);
            ////// And we can write out a few rows
            //Console.WriteLine($"Features column obtained post-transformation.");
            //foreach (var featureRow in featuresColumn)
            //    Console.WriteLine(string.Join(" ", featureRow.Features));

            IDataView testData = context.Data.LoadFromTextFile<Model3>(
                "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\test.csv",
                hasHeader: true);

            //Run the model on the test data set.
            var transformedTestData = pipeline.Fit(testData).Transform(testData);
            var evaluate = model.Transform(transformedTestData);

            // Convert IDataView object to a list.
            var predictions = context.Data.CreateEnumerable<Prediction>(
                evaluate, reuseRowObject: false).ToList();
            Console.WriteLine($"Label: {predictions[0].Label:F3}");
            Console.WriteLine($"Element: {predictions[0].Element:F3}");
        }
    }
}
