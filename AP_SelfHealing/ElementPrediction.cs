using AP_SelfHealing.MachineLearning.DataModels;
using Microsoft.ML;

namespace AP_SelfHealing
{
    public class ElementPrediction
    {
        public static MLContext context = new MLContext();
        PredictionEngine<ElementDetails, PredictedElement> _predEngine;
        ITransformer _trainedModel;
        IDataView trainData;

        private static string? _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _modelPath => "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Models\\models.zip";

        public void LoadAndPrepareData()
        {
            Console.WriteLine("----Path-----");
            Console.WriteLine(Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]));
            trainData = context.Data.LoadFromTextFile<ElementDetails>(
                "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\file.csv",
                hasHeader: true);

            var pipeline = ProcessData();

            var trainingPipeline = BuildAndTrainModel(trainData, pipeline);

            Evaluate(trainData.Schema);

            PredictIssue();
        }

        public IEstimator<ITransformer> ProcessData() 
        {
            var pipeline = context.Transforms.Conversion.MapValueToKey(inputColumnName: "Element", outputColumnName: "Label")
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "ControlId", outputColumnName: "ControlIdFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "CSSClass", outputColumnName: "CSSClassFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Value", outputColumnName: "ValueFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Role", outputColumnName: "RoleFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Type", outputColumnName: "TypeFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(context.Transforms.Text.FeaturizeText(inputColumnName: "Href", outputColumnName: "HrefFeaturized"))
                .Append(context.Transforms.Concatenate("Features", "ControlIdFeaturized", "NameFeaturized", "CSSClassFeaturized"
                    , "ValueFeaturized", "RoleFeaturized", "TypeFeaturized", "TitleFeaturized", "HrefFeaturized"))
                .AppendCacheCheckpoint(context);
                
            return pipeline;
        }

        public IEstimator<ITransformer> BuildAndTrainModel(IDataView trainigDataView,IEstimator<ITransformer> pipeline)
        {
            var trainingPipeline = pipeline.Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(trainigDataView);

            _predEngine = context.Model.CreatePredictionEngine<ElementDetails, PredictedElement>(_trainedModel);

            ElementDetails details = new ElementDetails
            {
                CSSClass = string.Empty,
                ControlId = "test",
                Href = string.Empty,
                Name = string.Empty,
                Role = string.Empty,
                Title = string.Empty,
                Type = string.Empty,
                Value = string.Empty
            };
            var predicition = _predEngine.Predict(details);

            Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {predicition.Element} ===============");

            return trainingPipeline;

        }

        void Evaluate(DataViewSchema trainingDataViewSchema)
        {
            var testData = context.Data.LoadFromTextFile<ElementDetails>(
                "C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\test.csv",
                hasHeader: true);
            var testMetrics = context.MulticlassClassification.Evaluate(_trainedModel.Transform(testData));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy:0.###}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy:0.###}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss:#.###}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction:#.###}");
            Console.WriteLine($"*************************************************************************************************************");

            SaveModelAsFile(context, trainingDataViewSchema, _trainedModel);
        }

        void SaveModelAsFile(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
        {
            context.Model.Save(model, trainingDataViewSchema, _modelPath);
        }

        void PredictIssue()
        {
            ITransformer loadedModel = context.Model.Load(_modelPath, out var modelInputSchema);

            ElementDetails details = new ElementDetails
            {
                CSSClass = string.Empty,
                ControlId = "Changed",
                Href = string.Empty,
                Name = "session_key",
                Role = string.Empty,
                Title = string.Empty,
                Type = "text",
                Value = string.Empty
            };

            _predEngine = context.Model.CreatePredictionEngine<ElementDetails, PredictedElement>(loadedModel);

            var prediction = _predEngine.Predict(details);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Element} ===============");
        }
    }
}
