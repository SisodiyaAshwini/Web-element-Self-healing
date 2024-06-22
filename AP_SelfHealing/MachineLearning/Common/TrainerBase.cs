using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using AP_SelfHealing.MachineLearning.DataModels;

namespace AP_SelfHealing.MachineLearning.Common
{
    /// <summary>
    /// Base class for Trainers.
    /// This class exposes methods for training, evaluating and saving ML Models.
    /// Classes that inherit this class need to assing concrete model and name; and to implement data pre-processing.
    /// </summary>
    public abstract class TrainerBase<TParameters> : ITrainerBase
        where TParameters : class
    {
        public string Name { get; protected set; }

        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "regression.mdl");

        protected readonly MLContext MlContext;

        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<RegressionPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            MlContext = new MLContext(111);
        }

        /// <summary>
        /// Train model on defined data.
        /// </summary>
        /// <param name="trainingFileName"></param>
        public void Fit(string trainingFileName)
        {
            //if (!File.Exists(trainingFileName))
            //{
            //    throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
            //}

            //_dataSplit = LoadAndPrepareData(trainingFileName);
            _dataSplit = LoadAndPrepareData();
            var dataProcessPipeline = BuildDataProcessingPipeline();
            var trainingPipeline = dataProcessPipeline.Append(_model);

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        /// <summary>
        /// Evaluate trained model.
        /// </summary>
        /// <returns>RegressionMetrics object.</returns>
        public RegressionMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return MlContext.Regression.Evaluate(testSetTransform);
        }

        /// <summary>
        /// Save Model in the file.
        /// </summary>
        public void Save()
        {
            MlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        /// <summary>
        /// Feature engeneering and data pre-processing.
        /// </summary>
        /// <returns>Data Processing Pipeline.</returns>
        private EstimatorChain<NormalizingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = MlContext.Transforms.CopyColumns("Label", nameof(Model.Element))
                //.Append(MlContext.Transforms.Categorical.OneHotEncoding("RiverCoast"))
                .Append(MlContext.Transforms.Concatenate("Features",
                                                "Element",
                                                "Line",
                                                "LinePosition",
                                                "ControlId",
                                                "Name",
                                                "CCSClass",
                                                "Value",
                                                "Label",
                                                "Role",
                                                "Type",
                                                "TabIndex",
                                                "Placeholder",
                                                "Title",
                                                "Href"))
               .Append(MlContext.Transforms.NormalizeLogMeanVariance("Features", "Features"));
                //.AppendCacheCheckpoint(MlContext);

            return dataProcessPipeline;
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData()
        {
            //var trainingDataView = MlContext.Data.LoadFromTextFile<Model>(trainingFileName, hasHeader: true);
            var trainingDataView = MlContext.Data.LoadFromTextFile<Model>("C:\\_Projects\\_MachineLearning\\self-healing\\AP_SelfHealing\\AP_SelfHealing\\Data\\file.csv", hasHeader: true);
            return MlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}

