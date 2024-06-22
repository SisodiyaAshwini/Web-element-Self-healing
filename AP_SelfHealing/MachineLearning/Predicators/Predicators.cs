using AP_SelfHealing.MachineLearning.DataModels;
using Microsoft.ML;

namespace AP_SelfHealing.MachineLearning.Predicators
{
    /// <summary>
    /// Loads Model from the file and makes predictions.
    /// </summary>
    public class Predictor
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "file.csv");
        private readonly MLContext _mlContext;

        private ITransformer _model;

        public Predictor()
        {
            _mlContext = new MLContext(111);
        }

        /// <summary>
        /// Runs prediction on new data.
        /// </summary>
        /// <param name="newSample">New data sample.</param>
        /// <returns>BostonHousingPricePredictions object.</returns>
        public ModelPrediction Predict(Model newSample)
        {
            LoadModel();

            var predictionEngine = _mlContext.Model.CreatePredictionEngine<
            Model, ModelPrediction>(_model);

            return predictionEngine.Predict(newSample);
        }

        private void LoadModel()
        {
            if (!File.Exists(ModelPath))
            {
                throw new FileNotFoundException($"File {ModelPath} doesn't exist.");
            }

            using (var stream = new FileStream(ModelPath,
                               FileMode.Open,
                               FileAccess.Read,
                            FileShare.Read))
            {
                _model = _mlContext.Model.Load(stream, out _);
            }

            if (_model == null)
            {
                throw new Exception($"Failed to load Model");
            }
        }
    }
}
